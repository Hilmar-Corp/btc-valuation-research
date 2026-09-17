from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


MANIFEST_ROOT = Path("manifests")
OUTPUT_TABLE_ROOT = Path("outputs/tables")
OUTPUT_FIGURE_ROOT = Path("outputs/figures/stationarity")
OUTPUT_LOG_ROOT = Path("outputs/logs")
CONFIG_PATH = Path("configs/research.yaml")


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
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, dict):
        fail(
            f"{path} root must be a mapping"
        )

    return obj



def recover_config_by_sha256(
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

        historical = yaml.safe_load(
            result.stdout.decode(
                "utf-8"
            )
        )

        if not isinstance(
            historical,
            dict,
        ):
            fail(
                "Historical research config "
                "is not a mapping"
            )

        return commit, historical

    fail(
        "Unable to recover historical "
        "research config matching "
        "monthly manifest SHA-256"
    )


def verify_config_provenance(
    monthly_config_sha256: str,
    current_config: dict[str, Any],
) -> dict[str, Any]:
    current_sha256 = sha256_file(
        CONFIG_PATH
    )

    if (
        current_sha256
        == monthly_config_sha256
    ):
        return {
            "mode": "exact_file_hash",
            "monthly_config_sha256": (
                monthly_config_sha256
            ),
            "current_config_sha256": (
                current_sha256
            ),
            "semantic_base_equal": True,
            "allowed_extension": None,
        }

    commit, historical = (
        recover_config_by_sha256(
            monthly_config_sha256
        )
    )

    historical_base = dict(
        historical
    )

    current_base = dict(
        current_config
    )

    historical_stationarity = (
        historical_base.pop(
            "stationarity_tests",
            None,
        )
    )

    current_stationarity = (
        current_base.pop(
            "stationarity_tests",
            None,
        )
    )

    if (
        historical_base
        != current_base
    ):
        common_keys = (
            set(historical_base)
            & set(current_base)
        )

        changed_keys = sorted(
            key
            for key in common_keys
            if (
                historical_base[key]
                != current_base[key]
            )
        )

        added_keys = sorted(
            set(current_base)
            - set(historical_base)
        )

        removed_keys = sorted(
            set(historical_base)
            - set(current_base)
        )

        fail(
            "Research config changed in "
            "panel-relevant protocol sections | "
            f"changed={changed_keys} | "
            f"added={added_keys} | "
            f"removed={removed_keys}"
        )

    if (
        historical_stationarity
        is not None
    ):
        fail(
            "Historical panel-build config "
            "already contained "
            "stationarity_tests"
        )

    if (
        current_stationarity
        is None
    ):
        fail(
            "Current stationarity_tests "
            "specification is missing"
        )

    return {
        "mode": (
            "semantic_protocol_extension"
        ),
        "monthly_config_sha256": (
            monthly_config_sha256
        ),
        "current_config_sha256": (
            current_sha256
        ),
        "historical_config_commit": (
            commit
        ),
        "semantic_base_equal": True,
        "allowed_extension": (
            "stationarity_tests"
        ),
    }


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


def adf_test(
    values: pd.Series,
    regression: str,
    autolag: str,
) -> dict[str, Any]:
    x = (
        values
        .dropna()
        .astype(float)
        .to_numpy()
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter("always")

        result = adfuller(
            x,
            regression=regression,
            autolag=autolag,
        )

    statistic = float(result[0])
    pvalue = float(result[1])
    usedlag = int(result[2])
    nobs = int(result[3])
    critical = {
        key: float(value)
        for key, value
        in result[4].items()
    }

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "used_lag": usedlag,
        "nobs": nobs,
        "critical_values": critical,
        "warnings": [
            str(w.message)
            for w in caught
        ],
    }


def kpss_test(
    values: pd.Series,
    regression: str,
    nlags: str,
) -> dict[str, Any]:
    x = (
        values
        .dropna()
        .astype(float)
        .to_numpy()
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter("always")

        result = kpss(
            x,
            regression=regression,
            nlags=nlags,
        )

    statistic = float(result[0])
    pvalue = float(result[1])
    usedlag = int(result[2])
    critical = {
        key: float(value)
        for key, value
        in result[3].items()
    }

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "used_lag": usedlag,
        "critical_values": critical,
        "warnings": [
            str(w.message)
            for w in caught
        ],
    }


def classify(
    level_adf_p: float,
    level_kpss_p: float,
    diff_adf_p: float,
    diff_kpss_p: float,
    alpha: float,
) -> str:
    level_adf_reject = (
        level_adf_p < alpha
    )

    level_kpss_reject = (
        level_kpss_p < alpha
    )

    diff_adf_reject = (
        diff_adf_p < alpha
    )

    diff_kpss_reject = (
        diff_kpss_p < alpha
    )

    if (
        level_adf_reject
        and not level_kpss_reject
    ):
        return "compatible_I0"

    if (
        not level_adf_reject
        and level_kpss_reject
        and diff_adf_reject
        and not diff_kpss_reject
    ):
        return "compatible_I1"

    return "inconclusive"


def safe_lags(
    nobs: int,
    requested: int,
) -> int:
    pacf_max = max(
        1,
        (nobs // 2) - 1,
    )

    return min(
        requested,
        pacf_max,
    )


def make_plots(
    dates: pd.Series,
    values: pd.Series,
    series_name: str,
    max_lags: int,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean = pd.DataFrame(
        {
            "date": dates,
            "value": values,
        }
    ).dropna()

    clean = clean.reset_index(
        drop=True
    )

    paths = []

    level_path = (
        output_dir
        / f"{series_name}_level.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        clean["date"],
        clean["value"],
    )

    ax.set_title(
        f"{series_name} — level"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(series_name)
    fig.tight_layout()
    fig.savefig(
        level_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        str(level_path)
    )

    diff = clean[
        "value"
    ].diff().dropna()

    diff_dates = clean[
        "date"
    ].iloc[1:]

    diff_path = (
        output_dir
        / f"{series_name}_diff.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        diff_dates,
        diff,
    )

    ax.set_title(
        f"{series_name} — first difference"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(
        f"Δ {series_name}"
    )
    fig.tight_layout()
    fig.savefig(
        diff_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        str(diff_path)
    )

    lags_level = safe_lags(
        len(clean),
        max_lags,
    )

    acf_level_path = (
        output_dir
        / f"{series_name}_acf_level.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    plot_acf(
        clean["value"],
        lags=lags_level,
        ax=ax,
        zero=False,
    )

    ax.set_title(
        f"{series_name} — ACF level"
    )
    fig.tight_layout()
    fig.savefig(
        acf_level_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        str(acf_level_path)
    )

    pacf_level_path = (
        output_dir
        / f"{series_name}_pacf_level.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    plot_pacf(
        clean["value"],
        lags=lags_level,
        ax=ax,
        zero=False,
        method="ywm",
    )

    ax.set_title(
        f"{series_name} — PACF level"
    )
    fig.tight_layout()
    fig.savefig(
        pacf_level_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        str(pacf_level_path)
    )

    lags_diff = safe_lags(
        len(diff),
        max_lags,
    )

    acf_diff_path = (
        output_dir
        / f"{series_name}_acf_diff.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    plot_acf(
        diff,
        lags=lags_diff,
        ax=ax,
        zero=False,
    )

    ax.set_title(
        f"{series_name} — ACF first difference"
    )
    fig.tight_layout()
    fig.savefig(
        acf_diff_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        str(acf_diff_path)
    )

    pacf_diff_path = (
        output_dir
        / f"{series_name}_pacf_diff.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    plot_pacf(
        diff,
        lags=lags_diff,
        ax=ax,
        zero=False,
        method="ywm",
    )

    ax.set_title(
        f"{series_name} — PACF first difference"
    )
    fig.tight_layout()
    fig.savefig(
        pacf_diff_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        str(pacf_diff_path)
    )

    return paths


def main() -> None:
    args = parse_args()

    cfg = load_yaml(
        CONFIG_PATH
    )

    spec = cfg[
        "stationarity_tests"
    ]

    alpha = float(
        spec["significance_level"]
    )

    manifest_path = (
        MANIFEST_ROOT
        / (
            f"monthly_panel_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not manifest_path.exists():
        fail(
            f"Missing monthly manifest: "
            f"{manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    if manifest.get(
        "status"
    ) != "PASS":
        fail(
            "Monthly panel manifest "
            "does not have PASS status"
        )

    panel_meta = manifest[
        "output"
    ]

    panel_path = Path(
        panel_meta["path"]
    )

    if not panel_path.exists():
        fail(
            f"Missing monthly panel: "
            f"{panel_path}"
        )

    if (
        sha256_file(panel_path)
        != panel_meta["sha256"]
    ):
        fail(
            "Monthly panel SHA-256 mismatch"
        )

    config_provenance = (
        verify_config_provenance(
            monthly_config_sha256=manifest[
                "research_config_sha256"
            ],
            current_config=cfg,
        )
    )

    df = pd.read_parquet(
        panel_path
    )

    df[
        "period_end_utc"
    ] = pd.to_datetime(
        df["period_end_utc"],
        utc=True,
    )

    rows = []
    detailed = {}
    all_plots = []

    for series_name in spec["series"]:
        if series_name not in df.columns:
            fail(
                f"Series missing from panel: "
                f"{series_name}"
            )

        level = (
            df[
                series_name
            ]
            .dropna()
            .astype(float)
        )

        diff = level.diff().dropna()

        if len(level) < 30:
            fail(
                f"{series_name}: "
                "too few observations"
            )

        level_adf = adf_test(
            level,
            regression=spec[
                "levels"
            ]["adf"]["regression"],
            autolag=spec[
                "levels"
            ]["adf"]["autolag"],
        )

        level_kpss = kpss_test(
            level,
            regression=spec[
                "levels"
            ]["kpss"]["regression"],
            nlags=spec[
                "levels"
            ]["kpss"]["nlags"],
        )

        diff_adf = adf_test(
            diff,
            regression=spec[
                "first_differences"
            ]["adf"]["regression"],
            autolag=spec[
                "first_differences"
            ]["adf"]["autolag"],
        )

        diff_kpss = kpss_test(
            diff,
            regression=spec[
                "first_differences"
            ]["kpss"]["regression"],
            nlags=spec[
                "first_differences"
            ]["kpss"]["nlags"],
        )

        classification = classify(
            level_adf[
                "pvalue"
            ],
            level_kpss[
                "pvalue"
            ],
            diff_adf[
                "pvalue"
            ],
            diff_kpss[
                "pvalue"
            ],
            alpha,
        )

        rows.append(
            {
                "series": series_name,
                "n_level": len(level),
                "n_diff": len(diff),
                "level_adf_stat": (
                    level_adf[
                        "statistic"
                    ]
                ),
                "level_adf_p": (
                    level_adf[
                        "pvalue"
                    ]
                ),
                "level_adf_lag": (
                    level_adf[
                        "used_lag"
                    ]
                ),
                "level_kpss_stat": (
                    level_kpss[
                        "statistic"
                    ]
                ),
                "level_kpss_p": (
                    level_kpss[
                        "pvalue"
                    ]
                ),
                "level_kpss_lag": (
                    level_kpss[
                        "used_lag"
                    ]
                ),
                "diff_adf_stat": (
                    diff_adf[
                        "statistic"
                    ]
                ),
                "diff_adf_p": (
                    diff_adf[
                        "pvalue"
                    ]
                ),
                "diff_adf_lag": (
                    diff_adf[
                        "used_lag"
                    ]
                ),
                "diff_kpss_stat": (
                    diff_kpss[
                        "statistic"
                    ]
                ),
                "diff_kpss_p": (
                    diff_kpss[
                        "pvalue"
                    ]
                ),
                "diff_kpss_lag": (
                    diff_kpss[
                        "used_lag"
                    ]
                ),
                "classification": (
                    classification
                ),
            }
        )

        detailed[
            series_name
        ] = {
            "n_level": len(level),
            "n_diff": len(diff),
            "level": {
                "adf": level_adf,
                "kpss": level_kpss,
            },
            "first_difference": {
                "adf": diff_adf,
                "kpss": diff_kpss,
            },
            "classification": (
                classification
            ),
        }

        dates = df.loc[
            df[
                series_name
            ].notna(),
            "period_end_utc",
        ]

        plots = make_plots(
            dates=dates,
            values=level,
            series_name=series_name,
            max_lags=int(
                spec[
                    "acf_pacf"
                ]["max_lags"]
            ),
            output_dir=(
                OUTPUT_FIGURE_ROOT
            ),
        )

        all_plots.extend(
            plots
        )

    results = pd.DataFrame(
        rows
    )

    OUTPUT_TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_TABLE_ROOT
        / "stationarity_results.csv"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    json_path = (
        OUTPUT_LOG_ROOT
        / "stationarity_diagnostics.json"
    )

    OUTPUT_LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "alpha": alpha,
        "monthly_panel": (
            str(panel_path)
        ),
        "monthly_panel_sha256": (
            sha256_file(
                panel_path
            )
        ),
        "research_config": (
            str(CONFIG_PATH)
        ),
        "research_config_sha256": (
            sha256_file(
                CONFIG_PATH
            )
        ),
        "config_provenance": (
            config_provenance
        ),
        "results": detailed,
        "plot_files": (
            all_plots
        ),
        "status": "PASS",
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    diagnostic_manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "base_run_id": (
            args.base_run_id
        ),
        "external_run_id": (
            args.external_run_id
        ),
        "monthly_manifest": (
            str(manifest_path)
        ),
        "monthly_manifest_sha256": (
            sha256_file(
                manifest_path
            )
        ),
        "research_config_sha256": (
            sha256_file(
                CONFIG_PATH
            )
        ),
        "config_provenance": (
            config_provenance
        ),
        "diagnostic_script": (
            str(Path(__file__))
        ),
        "diagnostic_script_sha256": (
            sha256_file(
                Path(__file__)
            )
        ),
        "results_csv": (
            str(csv_path)
        ),
        "results_csv_sha256": (
            sha256_file(
                csv_path
            )
        ),
        "results_json": (
            str(json_path)
        ),
        "results_json_sha256": (
            sha256_file(
                json_path
            )
        ),
        "plot_count": len(
            all_plots
        ),
    }

    diagnostic_manifest_path = (
        MANIFEST_ROOT
        / (
            "stationarity_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if diagnostic_manifest_path.exists():
        fail(
            f"Refusing to overwrite: "
            f"{diagnostic_manifest_path}"
        )

    diagnostic_manifest_path.write_text(
        json.dumps(
            diagnostic_manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "===== STATIONARITY DIAGNOSTICS ====="
    )

    for row in rows:
        print(
            row["series"],
            "| n=",
            row["n_level"],
            "| ADF(level)=",
            f"{row['level_adf_p']:.4g}",
            "| KPSS(level)=",
            f"{row['level_kpss_p']:.4g}",
            "| ADF(diff)=",
            f"{row['diff_adf_p']:.4g}",
            "| KPSS(diff)=",
            f"{row['diff_kpss_p']:.4g}",
            "|",
            row["classification"],
        )

    print()
    print(
        f"Table: {csv_path}"
    )
    print(
        f"Detailed results: {json_path}"
    )
    print(
        f"Figures: {OUTPUT_FIGURE_ROOT}"
    )
    print(
        f"Manifest: "
        f"{diagnostic_manifest_path}"
    )


if __name__ == "__main__":
    main()
