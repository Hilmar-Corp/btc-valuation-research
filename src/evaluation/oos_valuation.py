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

        candidate_sha = hashlib.sha256(
            result.stdout
        ).hexdigest()

        if candidate_sha != target_sha256:
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
        "oos_valuation",
        None,
    )

    new_extension = new.pop(
        "oos_valuation",
        None,
    )

    if old_extension is not None:
        fail(
            "Upstream config already contained "
            "oos_valuation"
        )

    if new_extension is None:
        fail(
            "Current config lacks oos_valuation"
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
            "Protocol drift outside OOS extension | "
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
        "allowed_extension": "oos_valuation",
        "semantic_base_equal": True,
    }


def metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = (
        actual
        - predicted
    )

    return {
        "n": int(
            len(error)
        ),
        "MAE": float(
            np.mean(
                np.abs(error)
            )
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    np.square(error)
                )
            )
        ),
        "bias": float(
            np.mean(error)
        ),
        "MedAE": float(
            np.median(
                np.abs(error)
            )
        ),
    }


def fit_expanding_model(
    df: pd.DataFrame,
    model_name: str,
    spec: dict[str, Any],
    minimum_training: int,
) -> pd.DataFrame:
    y_name = spec["dependent"]
    x_name = spec["explanatory"]
    transform = spec[
        "output_transform"
    ]

    rows = []

    for target_index in range(
        len(df)
    ):
        target = df.iloc[
            target_index
        ]

        if pd.isna(
            target[x_name]
        ):
            continue

        training = (
            df.iloc[
                :target_index
            ][
                [
                    "month",
                    y_name,
                    x_name,
                ]
            ]
            .dropna()
            .copy()
        )

        if (
            len(training)
            < minimum_training
        ):
            continue

        X = sm.add_constant(
            training[x_name]
        )

        result = sm.OLS(
            training[y_name],
            X,
        ).fit()

        alpha = float(
            result.params["const"]
        )

        beta = float(
            result.params[x_name]
        )

        x_t = float(
            target[x_name]
        )

        predicted_target = (
            alpha
            + beta * x_t
        )

        if (
            transform
            == "direct_log_price"
        ):
            predicted_log_price = (
                predicted_target
            )

        elif (
            transform
            == "log_market_cap_minus_log_supply"
        ):
            supply_t = float(
                target[
                    "btc_supply_current"
                ]
            )

            if supply_t <= 0:
                fail(
                    "Non-positive BTC supply"
                )

            predicted_log_price = (
                predicted_target
                - math.log(
                    supply_t
                )
            )

        else:
            fail(
                "Unknown output transform: "
                f"{transform}"
            )

        if target_index < 1:
            fail(
                "No previous-price baseline "
                "available"
            )

        previous_log_price = float(
            df.iloc[
                target_index - 1
            ][
                "log_btc_price"
            ]
        )

        expanding_mean = float(
            df.iloc[
                :target_index
            ][
                "log_btc_price"
            ].mean()
        )

        actual_log_price = float(
            target[
                "log_btc_price"
            ]
        )

        rows.append(
            {
                "model": model_name,
                "month": target["month"],
                "period_end_utc": (
                    target[
                        "period_end_utc"
                    ]
                ),
                "training_n": int(
                    len(training)
                ),
                "training_start_month": (
                    training[
                        "month"
                    ].iloc[0]
                ),
                "training_end_month": (
                    training[
                        "month"
                    ].iloc[-1]
                ),
                "alpha": alpha,
                "beta": beta,
                "x_t": x_t,
                "actual_log_price": (
                    actual_log_price
                ),
                "model_log_price": (
                    predicted_log_price
                ),
                "previous_price_log": (
                    previous_log_price
                ),
                "expanding_mean_log": (
                    expanding_mean
                ),
                "actual_price_usd": float(
                    math.exp(
                        actual_log_price
                    )
                ),
                "model_price_usd": float(
                    math.exp(
                        predicted_log_price
                    )
                ),
                "previous_price_usd": float(
                    math.exp(
                        previous_log_price
                    )
                ),
                "expanding_mean_price_usd": float(
                    math.exp(
                        expanding_mean
                    )
                ),
                "model_error_log": (
                    actual_log_price
                    - predicted_log_price
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_expected_sample(
    forecasts: pd.DataFrame,
    spec: dict[str, Any],
    model_name: str,
) -> None:
    expected_n = int(
        spec[
            "expected_oos_observations"
        ]
    )

    expected_first = spec[
        "expected_first_target_month"
    ]

    if len(forecasts) != expected_n:
        fail(
            f"{model_name}: expected "
            f"{expected_n} OOS rows, "
            f"found {len(forecasts)}"
        )

    if (
        forecasts[
            "month"
        ].iloc[0]
        != expected_first
    ):
        fail(
            f"{model_name}: first target "
            f"month mismatch"
        )


def evaluate_model_specific(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model_name, group in forecasts.groupby(
        "model",
        sort=True,
    ):
        definitions = {
            model_name: (
                "model_log_price"
            ),
            "previous_price": (
                "previous_price_log"
            ),
            "expanding_log_mean": (
                "expanding_mean_log"
            ),
        }

        for estimator, column in (
            definitions.items()
        ):
            result = metrics(
                group[
                    "actual_log_price"
                ],
                group[column],
            )

            rows.append(
                {
                    "reference_model": (
                        model_name
                    ),
                    "estimator": (
                        estimator
                    ),
                    "first_month": (
                        group[
                            "month"
                        ].iloc[0]
                    ),
                    "last_month": (
                        group[
                            "month"
                        ].iloc[-1]
                    ),
                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


def evaluate_common(
    forecasts: pd.DataFrame,
    expected_n: int,
    expected_first: str,
    expected_last: str,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    month_sets = [
        set(group["month"])
        for _, group
        in forecasts.groupby(
            "model"
        )
    ]

    common_months = sorted(
        set.intersection(
            *month_sets
        )
    )

    if len(common_months) != expected_n:
        fail(
            "Common OOS sample size mismatch"
        )

    if (
        common_months[0]
        != expected_first
        or common_months[-1]
        != expected_last
    ):
        fail(
            "Common OOS sample date mismatch"
        )

    common = forecasts[
        forecasts[
            "month"
        ].isin(
            common_months
        )
    ].copy()

    rows = []

    for model_name, group in common.groupby(
        "model",
        sort=True,
    ):
        result = metrics(
            group[
                "actual_log_price"
            ],
            group[
                "model_log_price"
            ],
        )

        rows.append(
            {
                "estimator": model_name,
                **result,
            }
        )

    baseline_source = (
        common[
            common["model"]
            == sorted(
                common[
                    "model"
                ].unique()
            )[0]
        ]
        .sort_values("month")
        .reset_index(drop=True)
    )

    for estimator, column in {
        "previous_price": (
            "previous_price_log"
        ),
        "expanding_log_mean": (
            "expanding_mean_log"
        ),
    }.items():
        result = metrics(
            baseline_source[
                "actual_log_price"
            ],
            baseline_source[column],
        )

        rows.append(
            {
                "estimator": estimator,
                **result,
            }
        )

    return (
        pd.DataFrame(rows),
        common_months,
    )


def main() -> None:
    args = parse_args()

    cfg = load_yaml(
        CONFIG_PATH
    )

    spec = cfg[
        "oos_valuation"
    ]

    structural_manifest_path = (
        MANIFEST_ROOT
        / (
            "structural_stability_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not structural_manifest_path.exists():
        fail(
            "Missing structural stability manifest"
        )

    structural_manifest = load_json(
        structural_manifest_path
    )

    if (
        structural_manifest.get(
            "status"
        )
        != "PASS"
    ):
        fail(
            "Structural stability manifest "
            "not PASS"
        )

    provenance = verify_protocol_extension(
        structural_manifest[
            "research_config_sha256"
        ],
        cfg,
    )

    descriptive_manifest_path = Path(
        structural_manifest[
            "upstream_descriptive_manifest"
        ]
    )

    descriptive_manifest = load_json(
        descriptive_manifest_path
    )

    stationarity_manifest_path = Path(
        descriptive_manifest[
            "upstream_stationarity_manifest"
        ]
    )

    stationarity_manifest = load_json(
        stationarity_manifest_path
    )

    monthly_manifest_path = Path(
        stationarity_manifest[
            "monthly_manifest"
        ]
    )

    monthly_manifest = load_json(
        monthly_manifest_path
    )

    panel_path = Path(
        monthly_manifest[
            "output"
        ]["path"]
    )

    if (
        sha256_file(panel_path)
        != monthly_manifest[
            "output"
        ]["sha256"]
    ):
        fail(
            "Monthly panel SHA-256 mismatch"
        )

    df = pd.read_parquet(
        panel_path
    )

    df[
        "period_end_utc"
    ] = pd.to_datetime(
        df[
            "period_end_utc"
        ],
        utc=True,
    )

    minimum_training = int(
        spec[
            "estimation"
        ][
            "minimum_training_observations"
        ]
    )

    forecast_frames = []

    for model_name, model_spec in (
        spec["models"].items()
    ):
        forecasts = (
            fit_expanding_model(
                df=df,
                model_name=model_name,
                spec=model_spec,
                minimum_training=(
                    minimum_training
                ),
            )
        )

        validate_expected_sample(
            forecasts,
            model_spec,
            model_name,
        )

        forecast_frames.append(
            forecasts
        )

    all_forecasts = pd.concat(
        forecast_frames,
        ignore_index=True,
    )

    specific_metrics = (
        evaluate_model_specific(
            all_forecasts
        )
    )

    common_metrics, common_months = (
        evaluate_common(
            all_forecasts,
            expected_n=int(
                spec[
                    "evaluation"
                ][
                    "expected_common_observations"
                ]
            ),
            expected_first=spec[
                "evaluation"
            ][
                "expected_common_first_month"
            ],
            expected_last=spec[
                "evaluation"
            ][
                "expected_common_last_month"
            ],
        )
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts_path = (
        TABLE_ROOT
        / "oos_valuation_forecasts.csv"
    )

    specific_path = (
        TABLE_ROOT
        / "oos_metrics_model_specific.csv"
    )

    common_path = (
        TABLE_ROOT
        / "oos_metrics_common.csv"
    )

    all_forecasts.to_csv(
        forecasts_path,
        index=False,
    )

    specific_metrics.to_csv(
        specific_path,
        index=False,
    )

    common_metrics.to_csv(
        common_path,
        index=False,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        LOG_ROOT
        / "oos_valuation.json"
    )

    details = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "interpretation": (
            "strict_expanding_"
            "pseudo_out_of_sample_valuation"
        ),
        "protocol_provenance": (
            provenance
        ),
        "minimum_training_observations": (
            minimum_training
        ),
        "common_sample": {
            "n": len(
                common_months
            ),
            "first_month": (
                common_months[0]
            ),
            "last_month": (
                common_months[-1]
            ),
        },
        "data_vintage_claim": (
            "not_realtime_vintage"
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
        "upstream_structural_manifest": (
            str(
                structural_manifest_path
            )
        ),
        "upstream_structural_manifest_sha256": (
            sha256_file(
                structural_manifest_path
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
        "forecasts": (
            str(
                forecasts_path
            )
        ),
        "forecasts_sha256": (
            sha256_file(
                forecasts_path
            )
        ),
        "model_specific_metrics": (
            str(
                specific_path
            )
        ),
        "model_specific_metrics_sha256": (
            sha256_file(
                specific_path
            )
        ),
        "common_metrics": (
            str(
                common_path
            )
        ),
        "common_metrics_sha256": (
            sha256_file(
                common_path
            )
        ),
        "details": (
            str(
                details_path
            )
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
            "oos_valuation_"
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
        "===== MODEL-SPECIFIC PSEUDO-OOS ====="
    )

    for _, row in specific_metrics.iterrows():
        print(
            row["reference_model"],
            "|",
            row["estimator"],
            "| n=",
            int(row["n"]),
            "| MAE=",
            f"{row['MAE']:.6f}",
            "| RMSE=",
            f"{row['RMSE']:.6f}",
            "| bias=",
            f"{row['bias']:.6f}",
            "| MedAE=",
            f"{row['MedAE']:.6f}",
        )

    print()
    print(
        "===== COMMON PSEUDO-OOS SAMPLE ====="
    )

    print(
        "months:",
        common_months[0],
        "->",
        common_months[-1],
        "| n=",
        len(common_months),
    )

    for _, row in common_metrics.iterrows():
        print(
            row["estimator"],
            "| MAE=",
            f"{row['MAE']:.6f}",
            "| RMSE=",
            f"{row['RMSE']:.6f}",
            "| bias=",
            f"{row['bias']:.6f}",
            "| MedAE=",
            f"{row['MedAE']:.6f}",
        )

    print()
    print(
        f"Forecasts: {forecasts_path}"
    )
    print(
        f"Model-specific metrics: {specific_path}"
    )
    print(
        f"Common metrics: {common_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
