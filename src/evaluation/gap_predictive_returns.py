from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from statsmodels.stats.multitest import multipletests


CONFIG_PATH = Path("configs/research.yaml")
MANIFEST_ROOT = Path("manifests")
TABLE_ROOT = Path("outputs/tables")
LOG_ROOT = Path("outputs/logs")


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(obj, dict):
        fail(
            f"{path}: YAML root must be mapping"
        )
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-run-id",
        required=True,
    )

    parser.add_argument(
        "--external-run-id",
        required=True,
    )

    return parser.parse_args()


def recover_config(
    target_sha256: str,
) -> tuple[str, dict[str, Any]]:
    commits = subprocess.check_output(
        [
            "git",
            "rev-list",
            "--all",
            "--",
            str(CONFIG_PATH),
        ],
        text=True,
    ).splitlines()

    for commit in commits:
        result = subprocess.run(
            [
                "git",
                "show",
                f"{commit}:{CONFIG_PATH}",
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            continue

        if (
            hashlib.sha256(
                result.stdout
            ).hexdigest()
            != target_sha256
        ):
            continue

        cfg = yaml.safe_load(
            result.stdout.decode("utf-8")
        )

        if not isinstance(cfg, dict):
            fail(
                "Historical config is not mapping"
            )

        return commit, cfg

    fail(
        "Unable to recover upstream "
        "research configuration"
    )


def verify_protocol_extension(
    upstream_sha256: str,
    current: dict[str, Any],
) -> dict[str, Any]:
    commit, historical = recover_config(
        upstream_sha256
    )

    old = dict(historical)
    new = dict(current)

    old_extension = old.pop(
        "gap_predictive_returns",
        None,
    )

    new_extension = new.pop(
        "gap_predictive_returns",
        None,
    )

    if old_extension is not None:
        fail(
            "Upstream config unexpectedly already "
            "contained gap_predictive_returns"
        )

    if new_extension is None:
        fail(
            "Current config lacks "
            "gap_predictive_returns"
        )

    if old != new:
        common = set(old) & set(new)

        changed = sorted(
            key
            for key in common
            if old[key] != new[key]
        )

        added = sorted(
            set(new) - set(old)
        )

        removed = sorted(
            set(old) - set(new)
        )

        fail(
            "Protocol drift outside predictive "
            "extension | "
            f"changed={changed} | "
            f"added={added} | "
            f"removed={removed}"
        )

    return {
        "mode": "semantic_protocol_extension",
        "historical_config_commit": commit,
        "upstream_config_sha256": (
            upstream_sha256
        ),
        "current_config_sha256": (
            sha256_file(CONFIG_PATH)
        ),
        "allowed_extension": (
            "gap_predictive_returns"
        ),
        "semantic_base_equal": True,
    }


def validate_monthly_sequence(
    group: pd.DataFrame,
    model: str,
) -> None:
    months = pd.PeriodIndex(
        group["month"],
        freq="M",
    )

    if len(months) < 2:
        fail(
            f"{model}: insufficient months"
        )

    expected = pd.period_range(
        months[0],
        months[-1],
        freq="M",
    )

    if not months.equals(expected):
        fail(
            f"{model}: forecast months "
            "are not contiguous"
        )


def build_sample(
    group: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    data = (
        group.sort_values("month")
        .reset_index(drop=True)
        .copy()
    )

    data["gap"] = (
        data["model_log_price"]
        - data["actual_log_price"]
    )

    data["forward_return"] = (
        data[
            "actual_log_price"
        ].shift(-horizon)
        - data[
            "actual_log_price"
        ]
    )

    return (
        data[
            [
                "month",
                "gap",
                "forward_return",
            ]
        ]
        .dropna()
        .reset_index(drop=True)
    )


def hac_regression(
    data: pd.DataFrame,
    horizon: int,
) -> dict[str, float]:
    y = data[
        "forward_return"
    ].to_numpy(
        dtype=float
    )

    x = data[
        "gap"
    ].to_numpy(
        dtype=float
    )

    X = sm.add_constant(
        x
    )

    result = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": horizon - 1,
        },
    )

    return {
        "alpha": float(
            result.params[0]
        ),
        "beta": float(
            result.params[1]
        ),
        "beta_hac_se": float(
            result.bse[1]
        ),
        "beta_hac_t": float(
            result.tvalues[1]
        ),
        "beta_hac_p": float(
            result.pvalues[1]
        ),
        "r_squared": float(
            result.rsquared
        ),
    }


def circular_block_bootstrap(
    data: pd.DataFrame,
    horizon: int,
    replications: int,
    seed: int,
) -> dict[str, float]:
    x = data[
        "gap"
    ].to_numpy(
        dtype=float
    )

    y = data[
        "forward_return"
    ].to_numpy(
        dtype=float
    )

    n = len(data)

    block_length = max(
        horizon,
        math.ceil(
            n ** (1 / 3)
        ),
    )

    blocks_needed = math.ceil(
        n / block_length
    )

    rng = np.random.default_rng(
        seed
    )

    betas = np.empty(
        replications,
        dtype=float,
    )

    offsets = np.arange(
        block_length
    )

    for r in range(
        replications
    ):
        starts = rng.integers(
            0,
            n,
            size=blocks_needed,
        )

        idx = np.concatenate(
            [
                (
                    start
                    + offsets
                )
                % n
                for start in starts
            ]
        )[:n]

        xb = x[idx]
        yb = y[idx]

        Xb = np.column_stack(
            [
                np.ones(n),
                xb,
            ]
        )

        params, _, rank, _ = (
            np.linalg.lstsq(
                Xb,
                yb,
                rcond=None,
            )
        )

        if rank < 2:
            fail(
                "Rank-deficient bootstrap sample"
            )

        betas[r] = float(
            params[1]
        )

    return {
        "block_length": int(
            block_length
        ),
        "bootstrap_beta_ci_low": float(
            np.quantile(
                betas,
                0.025,
            )
        ),
        "bootstrap_beta_ci_high": float(
            np.quantile(
                betas,
                0.975,
            )
        ),
        "bootstrap_probability_positive": float(
            np.mean(
                betas > 0
            )
        ),
    }


def infer(
    data: pd.DataFrame,
    horizon: int,
    replications: int,
    seed: int,
) -> dict[str, Any]:
    hac = hac_regression(
        data,
        horizon,
    )

    bootstrap = (
        circular_block_bootstrap(
            data=data,
            horizon=horizon,
            replications=replications,
            seed=seed,
        )
    )

    return {
        "n": int(
            len(data)
        ),
        "first_month": (
            data["month"].iloc[0]
        ),
        "last_month": (
            data["month"].iloc[-1]
        ),
        "horizon_months": int(
            horizon
        ),
        "hac_maxlags": int(
            horizon - 1
        ),
        **hac,
        **bootstrap,
    }


def main() -> None:
    args = parse_args()

    cfg = load_yaml(
        CONFIG_PATH
    )

    spec = cfg[
        "gap_predictive_returns"
    ]

    loss_manifest_path = (
        MANIFEST_ROOT
        / (
            "oos_loss_inference_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not loss_manifest_path.exists():
        fail(
            "Missing OOS loss inference manifest"
        )

    loss_manifest = load_json(
        loss_manifest_path
    )

    if (
        loss_manifest.get("status")
        != "PASS"
    ):
        fail(
            "OOS loss inference manifest "
            "is not PASS"
        )

    provenance = verify_protocol_extension(
        loss_manifest[
            "research_config_sha256"
        ],
        cfg,
    )

    oos_manifest_path = Path(
        loss_manifest[
            "upstream_oos_manifest"
        ]
    )

    oos_manifest = load_json(
        oos_manifest_path
    )

    forecasts_path = Path(
        oos_manifest[
            "forecasts"
        ]
    )

    if (
        sha256_file(
            forecasts_path
        )
        != oos_manifest[
            "forecasts_sha256"
        ]
    ):
        fail(
            "Frozen OOS forecasts "
            "SHA-256 mismatch"
        )

    forecasts = pd.read_csv(
        forecasts_path
    )

    required_columns = {
        "model",
        "month",
        "actual_log_price",
        "model_log_price",
    }

    missing = (
        required_columns
        - set(forecasts.columns)
    )

    if missing:
        fail(
            f"Missing forecast columns: "
            f"{sorted(missing)}"
        )

    models = [
        "scarcity",
        "network",
        "production_cost",
    ]

    horizons = [
        int(x)
        for x in spec[
            "forward_return"
        ]["horizons_months"]
    ]

    expected_primary = spec[
        "samples"
    ][
        "expected_primary_observations"
    ]

    expected_common = spec[
        "samples"
    ][
        "expected_common_observations"
    ]

    replications = int(
        spec[
            "block_bootstrap"
        ]["replications"]
    )

    base_seed = int(
        spec[
            "block_bootstrap"
        ]["random_seed"]
    )

    primary_rows = []
    sample_cache = {}

    for model_index, model in enumerate(
        models
    ):
        group = (
            forecasts[
                forecasts["model"]
                == model
            ]
            .sort_values("month")
            .reset_index(drop=True)
        )

        if group.empty:
            fail(
                f"Missing forecasts for {model}"
            )

        validate_monthly_sequence(
            group,
            model,
        )

        for horizon in horizons:
            data = build_sample(
                group,
                horizon,
            )

            expected_n = int(
                expected_primary[
                    model
                ][str(horizon)]
            )

            if len(data) != expected_n:
                fail(
                    f"{model} h={horizon}: "
                    f"expected {expected_n}, "
                    f"found {len(data)}"
                )

            sample_cache[
                (
                    model,
                    horizon,
                )
            ] = data

            result = infer(
                data=data,
                horizon=horizon,
                replications=replications,
                seed=(
                    base_seed
                    + 1000
                    * model_index
                    + horizon
                ),
            )

            primary_rows.append(
                {
                    "model": model,
                    **result,
                }
            )

    primary = pd.DataFrame(
        primary_rows
    )

    reject, adjusted, _, _ = (
        multipletests(
            primary[
                "beta_hac_p"
            ].to_numpy(),
            alpha=float(
                spec[
                    "multiplicity"
                ][
                    "primary_family"
                ]["alpha"]
            ),
            method="holm",
        )
    )

    primary[
        "beta_hac_p_holm"
    ] = adjusted

    primary[
        "beta_hac_reject_holm"
    ] = reject

    primary[
        "bootstrap_ci_excludes_zero"
    ] = (
        (
            primary[
                "bootstrap_beta_ci_low"
            ]
            > 0
        )
        |
        (
            primary[
                "bootstrap_beta_ci_high"
            ]
            < 0
        )
    )

    common_rows = []

    for horizon in horizons:
        common_months = sorted(
            set.intersection(
                *[
                    set(
                        sample_cache[
                            (
                                model,
                                horizon,
                            )
                        ]["month"]
                    )
                    for model in models
                ]
            )
        )

        expected_n = int(
            expected_common[
                str(horizon)
            ]
        )

        if len(common_months) != expected_n:
            fail(
                f"Common h={horizon}: "
                f"expected {expected_n}, "
                f"found {len(common_months)}"
            )

        for model_index, model in enumerate(
            models
        ):
            data = (
                sample_cache[
                    (
                        model,
                        horizon,
                    )
                ]
                .loc[
                    lambda x:
                    x["month"].isin(
                        common_months
                    )
                ]
                .sort_values("month")
                .reset_index(drop=True)
            )

            result = infer(
                data=data,
                horizon=horizon,
                replications=replications,
                seed=(
                    base_seed
                    + 10000
                    + 1000
                    * model_index
                    + horizon
                ),
            )

            common_rows.append(
                {
                    "model": model,
                    **result,
                }
            )

    common = pd.DataFrame(
        common_rows
    )

    common[
        "bootstrap_ci_excludes_zero"
    ] = (
        (
            common[
                "bootstrap_beta_ci_low"
            ]
            > 0
        )
        |
        (
            common[
                "bootstrap_beta_ci_high"
            ]
            < 0
        )
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_path = (
        TABLE_ROOT
        / "gap_predictive_returns_primary.csv"
    )

    common_path = (
        TABLE_ROOT
        / "gap_predictive_returns_common.csv"
    )

    primary.to_csv(
        primary_path,
        index=False,
    )

    common.to_csv(
        common_path,
        index=False,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        LOG_ROOT
        / "gap_predictive_returns.json"
    )

    details = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "protocol_provenance": (
            provenance
        ),
        "gap_source": (
            "frozen_expanding_pseudo_oos"
        ),
        "valuation_models_reestimated": (
            False
        ),
        "shared_current_price_component": (
            True
        ),
        "bootstrap_replications": (
            replications
        ),
    }

    details_path.write_text(
        json.dumps(
            details,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "base_run_id": (
            args.base_run_id
        ),
        "external_run_id": (
            args.external_run_id
        ),
        "upstream_loss_manifest": (
            str(loss_manifest_path)
        ),
        "upstream_loss_manifest_sha256": (
            sha256_file(
                loss_manifest_path
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
        "protocol_provenance": (
            provenance
        ),
        "script": str(
            Path(__file__)
        ),
        "script_sha256": (
            sha256_file(
                Path(__file__)
            )
        ),
        "primary_results": (
            str(primary_path)
        ),
        "primary_results_sha256": (
            sha256_file(
                primary_path
            )
        ),
        "common_results": (
            str(common_path)
        ),
        "common_results_sha256": (
            sha256_file(
                common_path
            )
        ),
        "details": (
            str(details_path)
        ),
        "details_sha256": (
            sha256_file(
                details_path
            )
        ),
    }

    manifest_path = (
        MANIFEST_ROOT
        / (
            "gap_predictive_returns_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if manifest_path.exists():
        fail(
            f"Refusing to overwrite: "
            f"{manifest_path}"
        )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "===== PRIMARY GAP -> FORWARD RETURNS ====="
    )

    for _, row in primary.iterrows():
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
            "  beta=",
            f"{row['beta']:.6f}",
            "| HAC SE=",
            f"{row['beta_hac_se']:.6f}",
            "| p=",
            f"{row['beta_hac_p']:.6g}",
            "| Holm=",
            f"{row['beta_hac_p_holm']:.6g}",
            "| reject=",
            bool(
                row[
                    "beta_hac_reject_holm"
                ]
            ),
            "| R2=",
            f"{row['r_squared']:.4f}",
        )

        print(
            "  bootstrap 95% beta CI=[",
            f"{row['bootstrap_beta_ci_low']:.6f}",
            ",",
            f"{row['bootstrap_beta_ci_high']:.6f}",
            "]",
            "| block=",
            int(
                row[
                    "block_length"
                ]
            ),
        )

    print()
    print(
        "===== COMMON-SAMPLE ROBUSTNESS ====="
    )

    for _, row in common.iterrows():
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
            "| beta=",
            f"{row['beta']:.6f}",
            "| HAC p=",
            f"{row['beta_hac_p']:.6g}",
            "| bootstrap CI=[",
            f"{row['bootstrap_beta_ci_low']:.6f}",
            ",",
            f"{row['bootstrap_beta_ci_high']:.6f}",
            "]",
        )

    print()
    print(
        f"Primary: {primary_path}"
    )
    print(
        f"Common: {common_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
