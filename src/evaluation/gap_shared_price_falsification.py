from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from statsmodels.stats.multitest import multipletests


CONFIG_PATH = Path("configs/research.yaml")
MANIFEST_ROOT = Path("manifests")
TABLE_ROOT = Path("outputs/tables")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-run-id",
        required=True,
    )

    p.add_argument(
        "--external-run-id",
        required=True,
    )

    return p.parse_args()


def fit_price_only(
    data: pd.DataFrame,
    horizon: int,
) -> dict:
    X = sm.add_constant(
        data[
            ["naive_gap"]
        ]
    )

    result = sm.OLS(
        data["forward_return"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": horizon - 1,
        },
    )

    return {
        "price_only_gamma": float(
            result.params["naive_gap"]
        ),
        "price_only_gamma_se": float(
            result.bse["naive_gap"]
        ),
        "price_only_gamma_p": float(
            result.pvalues["naive_gap"]
        ),
        "price_only_r2": float(
            result.rsquared
        ),
    }


def fit_incremental(
    data: pd.DataFrame,
    horizon: int,
) -> dict:
    X = sm.add_constant(
        data[
            [
                "naive_gap",
                "excess_model_gap",
            ]
        ]
    )

    result = sm.OLS(
        data["forward_return"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": horizon - 1,
        },
    )

    return {
        "gamma_naive": float(
            result.params[
                "naive_gap"
            ]
        ),
        "gamma_naive_p": float(
            result.pvalues[
                "naive_gap"
            ]
        ),
        "beta_excess": float(
            result.params[
                "excess_model_gap"
            ]
        ),
        "beta_excess_se": float(
            result.bse[
                "excess_model_gap"
            ]
        ),
        "beta_excess_p": float(
            result.pvalues[
                "excess_model_gap"
            ]
        ),
        "incremental_r2": float(
            result.rsquared
        ),
    }


def main() -> None:
    args = parse_args()

    cfg = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    spec = cfg[
        "gap_shared_price_falsification"
    ]

    q3_manifest_path = (
        MANIFEST_ROOT
        / (
            "gap_predictive_returns_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    q3_manifest = json.loads(
        q3_manifest_path.read_text(
            encoding="utf-8"
        )
    )

    forecasts_path = Path(
        q3_manifest[
            "frozen_oos_forecasts"
        ]
    )

    if (
        sha256_file(forecasts_path)
        != q3_manifest[
            "frozen_oos_forecasts_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen OOS forecast hash mismatch"
        )

    forecasts = pd.read_csv(
        forecasts_path
    )

    models = [
        "scarcity",
        "network",
        "production_cost",
    ]

    horizons = [
        3,
        6,
        12,
        24,
    ]

    rows = []

    for model in models:
        group = (
            forecasts[
                forecasts["model"]
                == model
            ]
            .sort_values("month")
            .reset_index(drop=True)
            .copy()
        )

        group["naive_gap"] = (
            group["previous_price_log"]
            - group["actual_log_price"]
        )

        group[
            "excess_model_gap"
        ] = (
            group["model_log_price"]
            - group["previous_price_log"]
        )

        for horizon in horizons:
            data = group.copy()

            data[
                "forward_return"
            ] = (
                data[
                    "actual_log_price"
                ].shift(
                    -horizon
                )
                - data[
                    "actual_log_price"
                ]
            )

            data = (
                data[
                    [
                        "month",
                        "forward_return",
                        "naive_gap",
                        "excess_model_gap",
                    ]
                ]
                .dropna()
                .reset_index(drop=True)
            )

            price_only = fit_price_only(
                data,
                horizon,
            )

            incremental = fit_incremental(
                data,
                horizon,
            )

            rows.append(
                {
                    "model": model,
                    "horizon_months": (
                        horizon
                    ),
                    "n": len(data),
                    **price_only,
                    **incremental,
                }
            )

    results = pd.DataFrame(
        rows
    )

    reject, adjusted, _, _ = (
        multipletests(
            results[
                "beta_excess_p"
            ].to_numpy(),
            alpha=float(
                spec[
                    "multiplicity"
                ][
                    "incremental_beta_family"
                ]["alpha"]
            ),
            method="holm",
        )
    )

    results[
        "beta_excess_p_holm"
    ] = adjusted

    results[
        "beta_excess_reject_holm"
    ] = reject

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_ROOT
        / "gap_shared_price_falsification.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "interpretation": (
            "post_primary_robustness_only"
        ),
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "upstream_q3_manifest": (
            str(q3_manifest_path)
        ),
        "upstream_q3_manifest_sha256": (
            sha256_file(
                q3_manifest_path
            )
        ),
        "frozen_oos_forecasts": (
            str(forecasts_path)
        ),
        "frozen_oos_forecasts_sha256": (
            sha256_file(
                forecasts_path
            )
        ),
        "research_config_sha256": (
            sha256_file(
                CONFIG_PATH
            )
        ),
        "script": str(
            Path(__file__)
        ),
        "script_sha256": (
            sha256_file(
                Path(__file__)
            )
        ),
        "results": str(
            output_path
        ),
        "results_sha256": (
            sha256_file(
                output_path
            )
        ),
    }

    manifest_path = (
        MANIFEST_ROOT
        / (
            "gap_shared_price_falsification_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if manifest_path.exists():
        raise RuntimeError(
            "Refusing to overwrite manifest"
        )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "===== SHARED-PRICE FALSIFICATION ====="
    )

    print(
        "beta_excess = incremental model-specific "
        "information beyond previous-price gap"
    )

    for _, row in results.iterrows():
        print()
        print(
            row["model"],
            "| h=",
            int(
                row[
                    "horizon_months"
                ]
            ),
            "| n=",
            int(row["n"]),
        )

        print(
            "  price-only:",
            "gamma=",
            f"{row['price_only_gamma']:.6f}",
            "| p=",
            f"{row['price_only_gamma_p']:.6g}",
            "| R2=",
            f"{row['price_only_r2']:.4f}",
        )

        print(
            "  incremental:",
            "beta_excess=",
            f"{row['beta_excess']:.6f}",
            "| SE=",
            f"{row['beta_excess_se']:.6f}",
            "| p=",
            f"{row['beta_excess_p']:.6g}",
            "| Holm=",
            f"{row['beta_excess_p_holm']:.6g}",
            "| reject=",
            bool(
                row[
                    "beta_excess_reject_holm"
                ]
            ),
            "| R2=",
            f"{row['incremental_r2']:.4f}",
        )

    print()
    print(
        f"Results: {output_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
