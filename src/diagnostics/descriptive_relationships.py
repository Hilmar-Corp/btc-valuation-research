from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from statsmodels.stats.diagnostic import acorr_ljungbox


CONFIG_PATH = Path("configs/research.yaml")
MANIFEST_ROOT = Path("manifests")
TABLE_ROOT = Path("outputs/tables")
FIGURE_ROOT = Path(
    "outputs/figures/descriptive_relationships"
)
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
        "Could not recover config matching "
        "upstream manifest SHA-256"
    )


def verify_protocol_extension(
    upstream_sha256: str,
    current: dict[str, Any],
) -> dict[str, Any]:
    current_sha = sha256_file(
        CONFIG_PATH
    )

    if current_sha == upstream_sha256:
        fail(
            "Current config contains no "
            "descriptive_relationships extension"
        )

    commit, historical = recover_config(
        upstream_sha256
    )

    old = dict(historical)
    new = dict(current)

    old_extension = old.pop(
        "descriptive_relationships",
        None,
    )

    new_extension = new.pop(
        "descriptive_relationships",
        None,
    )

    if old_extension is not None:
        fail(
            "Upstream config unexpectedly already "
            "contained descriptive_relationships"
        )

    if new_extension is None:
        fail(
            "Current config lacks "
            "descriptive_relationships"
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
            "Protocol drift outside permitted "
            "descriptive extension | "
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
            current_sha
        ),
        "allowed_extension": (
            "descriptive_relationships"
        ),
        "semantic_base_equal": True,
    }


def fit_ols_hac(
    y: pd.Series,
    x: pd.Series,
    maxlags: int,
) -> tuple[Any, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "y": y,
            "x": x,
        }
    ).dropna()

    X = sm.add_constant(
        data["x"]
    )

    model = sm.OLS(
        data["y"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
        },
    )

    return model, data


def model_summary(
    model: Any,
    data: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "nobs": int(model.nobs),
        "intercept": float(
            model.params["const"]
        ),
        "beta": float(
            model.params["x"]
        ),
        "intercept_se_hac": float(
            model.bse["const"]
        ),
        "beta_se_hac": float(
            model.bse["x"]
        ),
        "intercept_p_hac": float(
            model.pvalues["const"]
        ),
        "beta_p_hac": float(
            model.pvalues["x"]
        ),
        "r_squared": float(
            model.rsquared
        ),
        "adj_r_squared": float(
            model.rsquared_adj
        ),
        "residual_mean": float(
            model.resid.mean()
        ),
        "residual_std": float(
            model.resid.std(
                ddof=1
            )
        ),
        "x_min": float(
            data["x"].min()
        ),
        "x_max": float(
            data["x"].max()
        ),
        "y_min": float(
            data["y"].min()
        ),
        "y_max": float(
            data["y"].max()
        ),
    }


def ljung_box(
    residuals: pd.Series,
    lags: list[int],
) -> dict[str, Any]:
    out = acorr_ljungbox(
        residuals,
        lags=lags,
        return_df=True,
    )

    result = {}

    for lag in lags:
        result[str(lag)] = {
            "statistic": float(
                out.loc[
                    lag,
                    "lb_stat",
                ]
            ),
            "pvalue": float(
                out.loc[
                    lag,
                    "lb_pvalue",
                ]
            ),
        }

    return result


def rolling_beta(
    y: pd.Series,
    x: pd.Series,
    dates: pd.Series,
    window: int,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "date": dates,
            "y": y,
            "x": x,
        }
    ).dropna().reset_index(
        drop=True
    )

    rows = []

    for end in range(
        window,
        len(data) + 1,
    ):
        sample = data.iloc[
            end - window:end
        ]

        X = sm.add_constant(
            sample["x"]
        )

        result = sm.OLS(
            sample["y"],
            X,
        ).fit()

        rows.append(
            {
                "date": (
                    sample[
                        "date"
                    ].iloc[-1]
                ),
                "beta": float(
                    result.params["x"]
                ),
                "intercept": float(
                    result.params["const"]
                ),
                "r_squared": float(
                    result.rsquared
                ),
                "window": int(
                    window
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_figures(
    name: str,
    dates: pd.Series,
    y: pd.Series,
    x: pd.Series,
    level_model: Any,
    level_data: pd.DataFrame,
    rolling: pd.DataFrame,
) -> list[Path]:
    FIGURE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []

    scatter_path = (
        FIGURE_ROOT
        / f"{name}_scatter_level.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        level_data["x"],
        level_data["y"],
    )

    grid = np.linspace(
        level_data["x"].min(),
        level_data["x"].max(),
        200,
    )

    fitted_grid = (
        level_model.params["const"]
        + level_model.params["x"]
        * grid
    )

    ax.plot(
        grid,
        fitted_grid,
    )

    ax.set_title(
        f"{name} — descriptive log-level relation"
    )
    ax.set_xlabel("Explanatory variable")
    ax.set_ylabel("Dependent variable")
    fig.tight_layout()
    fig.savefig(
        scatter_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        scatter_path
    )

    paired = pd.DataFrame(
        {
            "date": dates,
            "y": y,
            "x": x,
        }
    ).dropna()

    X = sm.add_constant(
        paired["x"]
    )

    paired["fitted"] = (
        level_model.predict(X)
    )

    fitted_path = (
        FIGURE_ROOT
        / f"{name}_fitted_level.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        paired["date"],
        paired["y"],
        label="Observed",
    )

    ax.plot(
        paired["date"],
        paired["fitted"],
        label="OLS fitted",
    )

    ax.set_title(
        f"{name} — observed vs descriptive fitted level"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        fitted_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        fitted_path
    )

    residual_path = (
        FIGURE_ROOT
        / f"{name}_residual_level.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        paired["date"],
        level_model.resid,
    )

    ax.axhline(
        0.0,
    )

    ax.set_title(
        f"{name} — descriptive level residual"
    )
    fig.tight_layout()
    fig.savefig(
        residual_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        residual_path
    )

    rolling_path = (
        FIGURE_ROOT
        / f"{name}_rolling_beta.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        rolling["date"],
        rolling["beta"],
    )

    ax.set_title(
        f"{name} — rolling beta"
    )
    ax.set_ylabel("Beta")
    fig.tight_layout()
    fig.savefig(
        rolling_path,
        dpi=160,
    )
    plt.close(fig)

    paths.append(
        rolling_path
    )

    return paths


def main() -> None:
    args = parse_args()

    config = load_yaml(
        CONFIG_PATH
    )

    spec = config[
        "descriptive_relationships"
    ]

    stationarity_manifest_path = (
        MANIFEST_ROOT
        / (
            f"stationarity_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not stationarity_manifest_path.exists():
        fail(
            "Missing stationarity manifest"
        )

    stationarity_manifest = load_json(
        stationarity_manifest_path
    )

    if (
        stationarity_manifest.get(
            "status"
        )
        != "PASS"
    ):
        fail(
            "Stationarity diagnostics not PASS"
        )

    provenance = verify_protocol_extension(
        stationarity_manifest[
            "research_config_sha256"
        ],
        config,
    )

    stationarity_csv = Path(
        stationarity_manifest[
            "results_csv"
        ]
    )

    if (
        sha256_file(
            stationarity_csv
        )
        != stationarity_manifest[
            "results_csv_sha256"
        ]
    ):
        fail(
            "Stationarity results hash mismatch"
        )

    stationarity = pd.read_csv(
        stationarity_csv
    )

    expected_classifications = {
        "log_btc_price": "inconclusive",
        "log_btc_market_cap": "inconclusive",
        "log_active_addresses": "compatible_I1",
        "log_production_cost_electricity": "inconclusive",
        "log_stock_to_flow": "compatible_I0",
    }

    observed = dict(
        zip(
            stationarity["series"],
            stationarity[
                "classification"
            ],
        )
    )

    if observed != expected_classifications:
        fail(
            "Stationarity classifications differ "
            "from frozen admissibility state"
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
            "Monthly panel hash mismatch"
        )

    df = pd.read_parquet(
        panel_path
    )

    df["period_end_utc"] = pd.to_datetime(
        df["period_end_utc"],
        utc=True,
    )

    hac_maxlags = int(
        spec[
            "level_regression"
        ]["hac_maxlags"]
    )

    diff_hac_maxlags = int(
        spec[
            "first_difference_regression"
        ]["hac_maxlags"]
    )

    rolling_window = int(
        spec[
            "rolling_coefficients"
        ]["window_months"]
    )

    lb_lags = [
        int(x)
        for x in spec[
            "residual_diagnostics"
        ]["ljung_box_lags"]
    ]

    summary_rows = []
    detailed = {}
    all_figures = []

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, model_spec in spec[
        "models"
    ].items():
        y_name = model_spec[
            "dependent"
        ]

        x_name = model_spec[
            "explanatory"
        ]

        y = df[y_name]
        x = df[x_name]

        level_model, level_data = (
            fit_ols_hac(
                y,
                x,
                hac_maxlags,
            )
        )

        paired = pd.DataFrame(
            {
                "date": (
                    df[
                        "period_end_utc"
                    ]
                ),
                "y": y,
                "x": x,
            }
        ).dropna().reset_index(
            drop=True
        )

        diff_model, diff_data = (
            fit_ols_hac(
                paired["y"].diff(),
                paired["x"].diff(),
                diff_hac_maxlags,
            )
        )

        level_stats = model_summary(
            level_model,
            level_data,
        )

        diff_stats = model_summary(
            diff_model,
            diff_data,
        )

        level_lb = ljung_box(
            pd.Series(
                level_model.resid
            ),
            lb_lags,
        )

        diff_lb = ljung_box(
            pd.Series(
                diff_model.resid
            ),
            lb_lags,
        )

        rolling = rolling_beta(
            y=y,
            x=x,
            dates=df[
                "period_end_utc"
            ],
            window=rolling_window,
        )

        rolling_path = (
            TABLE_ROOT
            / (
                f"{name}_"
                "rolling_coefficients.csv"
            )
        )

        rolling.to_csv(
            rolling_path,
            index=False,
        )

        figure_paths = make_figures(
            name=name,
            dates=df[
                "period_end_utc"
            ],
            y=y,
            x=x,
            level_model=level_model,
            level_data=level_data,
            rolling=rolling,
        )

        all_figures.extend(
            figure_paths
        )

        summary_rows.append(
            {
                "model": name,
                "dependent": y_name,
                "explanatory": x_name,
                "n_level": (
                    level_stats["nobs"]
                ),
                "level_beta": (
                    level_stats["beta"]
                ),
                "level_beta_se_hac": (
                    level_stats[
                        "beta_se_hac"
                    ]
                ),
                "level_beta_p_hac": (
                    level_stats[
                        "beta_p_hac"
                    ]
                ),
                "level_r_squared": (
                    level_stats[
                        "r_squared"
                    ]
                ),
                "n_diff": (
                    diff_stats["nobs"]
                ),
                "diff_beta": (
                    diff_stats["beta"]
                ),
                "diff_beta_se_hac": (
                    diff_stats[
                        "beta_se_hac"
                    ]
                ),
                "diff_beta_p_hac": (
                    diff_stats[
                        "beta_p_hac"
                    ]
                ),
                "diff_r_squared": (
                    diff_stats[
                        "r_squared"
                    ]
                ),
                "rolling_beta_min": float(
                    rolling[
                        "beta"
                    ].min()
                ),
                "rolling_beta_median": float(
                    rolling[
                        "beta"
                    ].median()
                ),
                "rolling_beta_max": float(
                    rolling[
                        "beta"
                    ].max()
                ),
            }
        )

        detailed[name] = {
            "dependent": y_name,
            "explanatory": x_name,
            "interpretation": (
                "descriptive_only"
            ),
            "engle_granger_admissible": (
                False
            ),
            "level": {
                **level_stats,
                "ljung_box": (
                    level_lb
                ),
            },
            "first_difference": {
                **diff_stats,
                "ljung_box": (
                    diff_lb
                ),
            },
            "rolling": {
                "window_months": (
                    rolling_window
                ),
                "observations": int(
                    len(rolling)
                ),
                "beta_min": float(
                    rolling[
                        "beta"
                    ].min()
                ),
                "beta_median": float(
                    rolling[
                        "beta"
                    ].median()
                ),
                "beta_max": float(
                    rolling[
                        "beta"
                    ].max()
                ),
                "table": str(
                    rolling_path
                ),
                "table_sha256": (
                    sha256_file(
                        rolling_path
                    )
                ),
            },
            "figures": [
                str(path)
                for path in figure_paths
            ],
        }

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        TABLE_ROOT
        / "descriptive_relationships.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_path = (
        LOG_ROOT
        / "descriptive_relationships.json"
    )

    payload = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "interpretation": (
            "descriptive_only"
        ),
        "cointegration_claim_permitted": (
            False
        ),
        "protocol_provenance": (
            provenance
        ),
        "models": detailed,
    }

    detail_path.write_text(
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
        "upstream_stationarity_manifest": (
            str(
                stationarity_manifest_path
            )
        ),
        "upstream_stationarity_manifest_sha256": (
            sha256_file(
                stationarity_manifest_path
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
        "summary_table": (
            str(summary_path)
        ),
        "summary_table_sha256": (
            sha256_file(
                summary_path
            )
        ),
        "details": str(
            detail_path
        ),
        "details_sha256": (
            sha256_file(
                detail_path
            )
        ),
        "figures": [
            {
                "path": str(path),
                "sha256": (
                    sha256_file(path)
                ),
            }
            for path in all_figures
        ],
    }

    manifest_path = (
        MANIFEST_ROOT
        / (
            "descriptive_relationships_"
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
        "===== DESCRIPTIVE RELATIONSHIPS ====="
    )

    for row in summary_rows:
        print()
        print(
            row["model"]
        )
        print(
            "  level:",
            f"n={row['n_level']}",
            f"beta={row['level_beta']:.6f}",
            f"HAC_SE={row['level_beta_se_hac']:.6f}",
            f"p={row['level_beta_p_hac']:.6g}",
            f"R2={row['level_r_squared']:.4f}",
        )
        print(
            "  diff: ",
            f"n={row['n_diff']}",
            f"beta={row['diff_beta']:.6f}",
            f"HAC_SE={row['diff_beta_se_hac']:.6f}",
            f"p={row['diff_beta_p_hac']:.6g}",
            f"R2={row['diff_r_squared']:.4f}",
        )
        print(
            "  rolling beta:",
            f"min={row['rolling_beta_min']:.4f}",
            f"median={row['rolling_beta_median']:.4f}",
            f"max={row['rolling_beta_max']:.4f}",
        )

    print()
    print(
        f"Summary: {summary_path}"
    )
    print(
        f"Details: {detail_path}"
    )
    print(
        f"Figures: {FIGURE_ROOT}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
