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


def load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_yaml(
    path: Path,
) -> dict[str, Any]:
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
        "Unable to recover upstream config"
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
        "oos_comparison_inference",
        None,
    )

    new_extension = new.pop(
        "oos_comparison_inference",
        None,
    )

    if old_extension is not None:
        fail(
            "Upstream config already contained "
            "oos_comparison_inference"
        )

    if new_extension is None:
        fail(
            "Current config lacks "
            "oos_comparison_inference"
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
            "Protocol drift outside inference "
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
            "oos_comparison_inference"
        ),
        "semantic_base_equal": True,
    }


def hac_mean_test(
    differential: np.ndarray,
    maxlags: int,
) -> dict[str, float]:
    y = np.asarray(
        differential,
        dtype=float,
    )

    X = np.ones(
        (len(y), 1),
        dtype=float,
    )

    result = sm.OLS(
        y,
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
        },
    )

    return {
        "mean": float(
            result.params[0]
        ),
        "hac_se": float(
            result.bse[0]
        ),
        "t_stat": float(
            result.tvalues[0]
        ),
        "pvalue_two_sided": float(
            result.pvalues[0]
        ),
    }


def moving_block_bootstrap_means(
    values: np.ndarray,
    replications: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(
        values,
        dtype=float,
    )

    n = len(x)

    if block_length < 1:
        fail(
            "Block length must be positive"
        )

    if block_length > n:
        fail(
            "Block length exceeds sample"
        )

    starts = np.arange(
        0,
        n - block_length + 1,
    )

    blocks_needed = math.ceil(
        n / block_length
    )

    rng = np.random.default_rng(
        seed
    )

    out = np.empty(
        replications,
        dtype=float,
    )

    for r in range(
        replications
    ):
        selected = rng.choice(
            starts,
            size=blocks_needed,
            replace=True,
        )

        sample = np.concatenate(
            [
                x[
                    start:
                    start + block_length
                ]
                for start in selected
            ]
        )[:n]

        out[r] = float(
            np.mean(sample)
        )

    return out


def main() -> None:
    args = parse_args()

    cfg = load_yaml(
        CONFIG_PATH
    )

    spec = cfg[
        "oos_comparison_inference"
    ]

    oos_manifest_path = (
        MANIFEST_ROOT
        / (
            "oos_valuation_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not oos_manifest_path.exists():
        fail(
            "Missing OOS valuation manifest"
        )

    oos_manifest = load_json(
        oos_manifest_path
    )

    if (
        oos_manifest.get(
            "status"
        )
        != "PASS"
    ):
        fail(
            "OOS valuation manifest not PASS"
        )

    provenance = verify_protocol_extension(
        oos_manifest[
            "research_config_sha256"
        ],
        cfg,
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
            "OOS forecasts SHA-256 mismatch"
        )

    forecasts = pd.read_csv(
        forecasts_path
    )

    models = list(
        spec["models"]
    )

    sample_spec = spec[
        "sample"
    ]

    expected_n = int(
        sample_spec[
            "expected_observations"
        ]
    )

    expected_first = sample_spec[
        "expected_first_month"
    ]

    expected_last = sample_spec[
        "expected_last_month"
    ]

    month_sets = []

    for model in models:
        group = forecasts[
            forecasts["model"]
            == model
        ]

        if group.empty:
            fail(
                f"Missing model forecasts: {model}"
            )

        month_sets.append(
            set(group["month"])
        )

    common_months = sorted(
        set.intersection(
            *month_sets
        )
    )

    if len(common_months) != expected_n:
        fail(
            "Common sample count mismatch"
        )

    if (
        common_months[0]
        != expected_first
        or common_months[-1]
        != expected_last
    ):
        fail(
            "Common sample date mismatch"
        )

    bootstrap_spec = spec[
        "block_bootstrap"
    ]

    replications = int(
        bootstrap_spec[
            "replications"
        ]
    )

    seed = int(
        bootstrap_spec[
            "random_seed"
        ]
    )

    block_length = int(
        math.ceil(
            expected_n ** (1 / 3)
        )
    )

    if (
        block_length
        != int(
            bootstrap_spec[
                "expected_block_length_for_n_61"
            ]
        )
    ):
        fail(
            "Block-length rule mismatch"
        )

    confidence_level = float(
        bootstrap_spec[
            "confidence_level"
        ]
    )

    alpha = (
        1.0
        - confidence_level
    )

    hac_maxlags = int(
        spec[
            "hac_mean_test"
        ]["maxlags"]
    )

    rows = []

    for model_index, model in enumerate(
        models
    ):
        group = (
            forecasts[
                (
                    forecasts["model"]
                    == model
                )
                &
                forecasts[
                    "month"
                ].isin(
                    common_months
                )
            ]
            .sort_values("month")
            .reset_index(drop=True)
        )

        if len(group) != expected_n:
            fail(
                f"{model}: common sample mismatch"
            )

        actual = group[
            "actual_log_price"
        ].to_numpy(
            dtype=float
        )

        model_pred = group[
            "model_log_price"
        ].to_numpy(
            dtype=float
        )

        baseline_pred = group[
            "previous_price_log"
        ].to_numpy(
            dtype=float
        )

        model_error = (
            actual
            - model_pred
        )

        baseline_error = (
            actual
            - baseline_pred
        )

        losses = {
            "absolute_error": (
                np.abs(model_error),
                np.abs(baseline_error),
            ),
            "squared_error": (
                np.square(model_error),
                np.square(baseline_error),
            ),
        }

        for loss_index, (
            loss_name,
            (
                model_loss,
                baseline_loss,
            ),
        ) in enumerate(
            losses.items()
        ):
            differential = (
                model_loss
                - baseline_loss
            )

            hac = hac_mean_test(
                differential,
                hac_maxlags,
            )

            bootstrap = (
                moving_block_bootstrap_means(
                    differential,
                    replications=(
                        replications
                    ),
                    block_length=(
                        block_length
                    ),
                    seed=(
                        seed
                        + 100 * model_index
                        + loss_index
                    ),
                )
            )

            lower = float(
                np.quantile(
                    bootstrap,
                    alpha / 2,
                )
            )

            upper = float(
                np.quantile(
                    bootstrap,
                    1 - alpha / 2,
                )
            )

            bootstrap_probability_positive = (
                float(
                    np.mean(
                        bootstrap > 0
                    )
                )
            )

            rows.append(
                {
                    "model": model,
                    "loss": loss_name,
                    "n": expected_n,
                    "mean_model_loss": float(
                        np.mean(
                            model_loss
                        )
                    ),
                    "mean_baseline_loss": float(
                        np.mean(
                            baseline_loss
                        )
                    ),
                    "mean_loss_differential": (
                        float(
                            np.mean(
                                differential
                            )
                        )
                    ),
                    "hac_se": (
                        hac["hac_se"]
                    ),
                    "hac_t": (
                        hac["t_stat"]
                    ),
                    "hac_p_two_sided": (
                        hac[
                            "pvalue_two_sided"
                        ]
                    ),
                    "bootstrap_ci_low": (
                        lower
                    ),
                    "bootstrap_ci_high": (
                        upper
                    ),
                    "bootstrap_probability_positive": (
                        bootstrap_probability_positive
                    ),
                    "bootstrap_replications": (
                        replications
                    ),
                    "block_length": (
                        block_length
                    ),
                }
            )

    results = pd.DataFrame(
        rows
    )

    reject, adjusted, _, _ = (
        multipletests(
            results[
                "hac_p_two_sided"
            ].to_numpy(),
            alpha=float(
                spec[
                    "multiplicity"
                ]["alpha"]
            ),
            method="holm",
        )
    )

    results[
        "hac_p_holm"
    ] = adjusted

    results[
        "hac_reject_holm"
    ] = reject

    results[
        "bootstrap_ci_entirely_positive"
    ] = (
        results[
            "bootstrap_ci_low"
        ]
        > 0
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_ROOT
        / "oos_loss_differential_inference.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        LOG_ROOT
        / "oos_loss_differential_inference.json"
    )

    payload = {
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
        "common_sample": {
            "first_month": (
                common_months[0]
            ),
            "last_month": (
                common_months[-1]
            ),
            "n": len(
                common_months
            ),
        },
        "benchmark": "previous_price",
        "loss_differential_sign": (
            "positive_means_model_worse"
        ),
        "block_bootstrap": {
            "replications": (
                replications
            ),
            "block_length": (
                block_length
            ),
            "rule": "ceil(n^(1/3))",
        },
    }

    details_path.write_text(
        json.dumps(
            payload,
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
        "upstream_oos_manifest": (
            str(oos_manifest_path)
        ),
        "upstream_oos_manifest_sha256": (
            sha256_file(
                oos_manifest_path
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
        "results": (
            str(output_path)
        ),
        "results_sha256": (
            sha256_file(
                output_path
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
            "oos_loss_inference_"
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
        "===== OOS LOSS DIFFERENTIAL INFERENCE ====="
    )

    print(
        "Positive differential = model worse "
        "than previous_price"
    )

    print(
        "Common sample:",
        common_months[0],
        "->",
        common_months[-1],
        "| n=",
        len(common_months),
        "| block=",
        block_length,
    )

    for _, row in results.iterrows():
        print()
        print(
            row["model"],
            "|",
            row["loss"],
        )

        print(
            "  mean diff=",
            f"{row['mean_loss_differential']:.6f}",
            "| HAC p=",
            f"{row['hac_p_two_sided']:.6g}",
            "| Holm=",
            f"{row['hac_p_holm']:.6g}",
            "| reject=",
            bool(
                row[
                    "hac_reject_holm"
                ]
            ),
        )

        print(
            "  bootstrap 95% CI=[",
            f"{row['bootstrap_ci_low']:.6f}",
            ",",
            f"{row['bootstrap_ci_high']:.6f}",
            "]",
            "| entirely > 0=",
            bool(
                row[
                    "bootstrap_ci_entirely_positive"
                ]
            ),
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
