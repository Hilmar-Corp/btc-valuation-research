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
from scipy.stats import f as f_distribution
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
            f"{path}: root must be mapping"
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
        "Could not recover upstream "
        "research configuration"
    )


def verify_protocol_extension(
    upstream_sha256: str,
    current: dict[str, Any],
) -> dict[str, Any]:
    current_sha = sha256_file(
        CONFIG_PATH
    )

    commit, historical = recover_config(
        upstream_sha256
    )

    old = dict(historical)
    new = dict(current)

    old_extension = old.pop(
        "structural_stability",
        None,
    )

    new_extension = new.pop(
        "structural_stability",
        None,
    )

    if old_extension is not None:
        fail(
            "Upstream config already contained "
            "structural_stability"
        )

    if new_extension is None:
        fail(
            "Current config lacks "
            "structural_stability"
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
            "Protocol drift outside structural "
            "stability extension | "
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
            "structural_stability"
        ),
        "semantic_base_equal": True,
    }


def build_stage_frame(
    df: pd.DataFrame,
    y_name: str,
    x_name: str,
    stage: str,
) -> pd.DataFrame:
    data = df[
        [
            "period_end_utc",
            y_name,
            x_name,
        ]
    ].dropna().copy()

    data = data.rename(
        columns={
            y_name: "y",
            x_name: "x",
        }
    )

    data = data.sort_values(
        "period_end_utc"
    ).reset_index(
        drop=True
    )

    if stage == "level":
        return data

    if stage == "first_difference":
        data["y"] = data["y"].diff()
        data["x"] = data["x"].diff()

        return (
            data.dropna()
            .reset_index(drop=True)
        )

    fail(
        f"Unknown stage: {stage}"
    )


def rss_ols(
    data: pd.DataFrame,
) -> float:
    if len(data) < 3:
        return float("inf")

    X = sm.add_constant(
        data["x"]
    )

    result = sm.OLS(
        data["y"],
        X,
    ).fit()

    return float(
        np.sum(
            np.square(
                result.resid
            )
        )
    )


def chow_test(
    data: pd.DataFrame,
    break_month: str,
) -> dict[str, Any]:
    month = (
        data["period_end_utc"]
        .dt.to_period("M")
        .astype(str)
    )

    before = data[
        month < break_month
    ]

    after = data[
        month >= break_month
    ]

    k = 2

    if (
        len(before) <= k
        or len(after) <= k
    ):
        fail(
            f"Break {break_month}: "
            "insufficient observations"
        )

    rss_pooled = rss_ols(data)
    rss_before = rss_ols(before)
    rss_after = rss_ols(after)

    numerator = (
        rss_pooled
        - (
            rss_before
            + rss_after
        )
    ) / k

    denominator_df = (
        len(before)
        + len(after)
        - 2 * k
    )

    denominator = (
        rss_before
        + rss_after
    ) / denominator_df

    statistic = (
        numerator
        / denominator
    )

    pvalue = float(
        f_distribution.sf(
            statistic,
            k,
            denominator_df,
        )
    )

    return {
        "n_before": int(
            len(before)
        ),
        "n_after": int(
            len(after)
        ),
        "statistic": float(
            statistic
        ),
        "df1": k,
        "df2": int(
            denominator_df
        ),
        "pvalue": pvalue,
    }


def hac_break_wald(
    data: pd.DataFrame,
    break_month: str,
    maxlags: int,
) -> dict[str, Any]:
    work = data.copy()

    month = (
        work["period_end_utc"]
        .dt.to_period("M")
        .astype(str)
    )

    work["break_dummy"] = (
        month >= break_month
    ).astype(float)

    work["x_break"] = (
        work["x"]
        * work["break_dummy"]
    )

    X = sm.add_constant(
        work[
            [
                "x",
                "break_dummy",
                "x_break",
            ]
        ]
    )

    result = sm.OLS(
        work["y"],
        X,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
        },
    )

    restriction = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    test = result.wald_test(
        restriction,
        scalar=True,
    )

    return {
        "statistic": float(
            test.statistic
        ),
        "pvalue": float(
            test.pvalue
        ),
        "intercept_pre": float(
            result.params["const"]
        ),
        "beta_pre": float(
            result.params["x"]
        ),
        "intercept_shift": float(
            result.params[
                "break_dummy"
            ]
        ),
        "beta_shift": float(
            result.params[
                "x_break"
            ]
        ),
        "intercept_post": float(
            result.params["const"]
            + result.params[
                "break_dummy"
            ]
        ),
        "beta_post": float(
            result.params["x"]
            + result.params[
                "x_break"
            ]
        ),
    }


def segment_fit(
    data: pd.DataFrame,
    start: int,
    end: int,
) -> dict[str, float]:
    sample = data.iloc[
        start:end
    ]

    X = sm.add_constant(
        sample["x"]
    )

    result = sm.OLS(
        sample["y"],
        X,
    ).fit()

    rss = float(
        np.sum(
            np.square(
                result.resid
            )
        )
    )

    return {
        "rss": rss,
        "intercept": float(
            result.params["const"]
        ),
        "beta": float(
            result.params["x"]
        ),
    }


def detect_unknown_breaks(
    data: pd.DataFrame,
    max_breaks: int,
    min_segment: int,
) -> dict[str, Any]:
    n = len(data)

    max_segments = (
        max_breaks + 1
    )

    rss_cache: dict[
        tuple[int, int],
        float,
    ] = {}

    def cost(
        start: int,
        end: int,
    ) -> float:
        key = (
            start,
            end,
        )

        if key not in rss_cache:
            rss_cache[key] = (
                segment_fit(
                    data,
                    start,
                    end,
                )["rss"]
            )

        return rss_cache[key]

    inf = float("inf")

    dp = [
        [inf] * (n + 1)
        for _ in range(
            max_segments + 1
        )
    ]

    prev = [
        [None] * (n + 1)
        for _ in range(
            max_segments + 1
        )
    ]

    dp[0][0] = 0.0

    for segments in range(
        1,
        max_segments + 1,
    ):
        min_end = (
            segments
            * min_segment
        )

        for end in range(
            min_end,
            n + 1,
        ):
            min_start = (
                (segments - 1)
                * min_segment
            )

            max_start = (
                end
                - min_segment
            )

            for start in range(
                min_start,
                max_start + 1,
            ):
                prior = dp[
                    segments - 1
                ][start]

                if not math.isfinite(
                    prior
                ):
                    continue

                candidate = (
                    prior
                    + cost(
                        start,
                        end,
                    )
                )

                if candidate < dp[
                    segments
                ][end]:
                    dp[
                        segments
                    ][end] = candidate

                    prev[
                        segments
                    ][end] = start

    candidates = []

    for segments in range(
        1,
        max_segments + 1,
    ):
        rss = dp[
            segments
        ][n]

        if not math.isfinite(
            rss
        ):
            continue

        breaks = (
            segments - 1
        )

        k = (
            2 * segments
            + breaks
        )

        bic = (
            n
            * math.log(
                rss / n
            )
            + k
            * math.log(n)
        )

        candidates.append(
            {
                "segments": segments,
                "breaks": breaks,
                "rss": rss,
                "bic": bic,
            }
        )

    if not candidates:
        fail(
            "No admissible segmentation"
        )

    selected = min(
        candidates,
        key=lambda x: x["bic"],
    )

    segments = int(
        selected["segments"]
    )

    end = n
    boundaries = [n]

    for current_segments in range(
        segments,
        0,
        -1,
    ):
        start = prev[
            current_segments
        ][end]

        if start is None:
            if current_segments == 1:
                start = 0
            else:
                fail(
                    "Broken segmentation backtrack"
                )

        boundaries.append(
            start
        )

        end = start

    boundaries = sorted(
        set(boundaries)
    )

    if boundaries[0] != 0:
        boundaries.insert(
            0,
            0,
        )

    if boundaries[-1] != n:
        boundaries.append(
            n
        )

    detected_break_indices = (
        boundaries[1:-1]
    )

    detected_break_months = [
        data[
            "period_end_utc"
        ]
        .iloc[index]
        .to_period("M")
        .strftime("%Y-%m")
        for index
        in detected_break_indices
    ]

    segment_details = []

    for start, end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        fit = segment_fit(
            data,
            start,
            end,
        )

        segment_details.append(
            {
                "start_month": (
                    data[
                        "period_end_utc"
                    ]
                    .iloc[start]
                    .to_period("M")
                    .strftime("%Y-%m")
                ),
                "end_month": (
                    data[
                        "period_end_utc"
                    ]
                    .iloc[
                        end - 1
                    ]
                    .to_period("M")
                    .strftime("%Y-%m")
                ),
                "n": int(
                    end - start
                ),
                "intercept": (
                    fit[
                        "intercept"
                    ]
                ),
                "beta": (
                    fit["beta"]
                ),
                "rss": (
                    fit["rss"]
                ),
            }
        )

    return {
        "n": int(n),
        "selected_breaks": int(
            selected["breaks"]
        ),
        "selected_segments": int(
            selected["segments"]
        ),
        "selected_bic": float(
            selected["bic"]
        ),
        "selected_rss": float(
            selected["rss"]
        ),
        "break_months": (
            detected_break_months
        ),
        "segments": (
            segment_details
        ),
        "candidate_models": (
            candidates
        ),
    }


def main() -> None:
    args = parse_args()

    cfg = load_yaml(
        CONFIG_PATH
    )

    spec = cfg[
        "structural_stability"
    ]

    descriptive_manifest_path = (
        MANIFEST_ROOT
        / (
            "descriptive_relationships_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not descriptive_manifest_path.exists():
        fail(
            "Missing descriptive manifest"
        )

    descriptive_manifest = load_json(
        descriptive_manifest_path
    )

    if (
        descriptive_manifest.get(
            "status"
        )
        != "PASS"
    ):
        fail(
            "Descriptive manifest not PASS"
        )

    provenance = verify_protocol_extension(
        descriptive_manifest[
            "research_config_sha256"
        ],
        cfg,
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

    models = cfg[
        "descriptive_relationships"
    ]["models"]

    stages = spec["stages"]

    break_months = spec[
        "predefined_breaks"
    ]["months"]

    hac_maxlags = int(
        spec[
            "predefined_tests"
        ][
            "hac_interaction_wald"
        ]["hac_maxlags"]
    )

    max_breaks = int(
        spec[
            "unknown_breaks"
        ]["max_breaks"]
    )

    min_segment = int(
        spec[
            "unknown_breaks"
        ][
            "min_segment_months"
        ]
    )

    predefined_rows = []
    unknown_rows = []
    unknown_details = {}

    for model_name, model_spec in models.items():
        for stage in stages:
            data = build_stage_frame(
                df=df,
                y_name=model_spec[
                    "dependent"
                ],
                x_name=model_spec[
                    "explanatory"
                ],
                stage=stage,
            )

            for break_month in break_months:
                chow = chow_test(
                    data,
                    break_month,
                )

                robust = hac_break_wald(
                    data,
                    break_month,
                    hac_maxlags,
                )

                predefined_rows.append(
                    {
                        "model": model_name,
                        "stage": stage,
                        "break_month": (
                            break_month
                        ),
                        "n_before": (
                            chow["n_before"]
                        ),
                        "n_after": (
                            chow["n_after"]
                        ),
                        "chow_f": (
                            chow["statistic"]
                        ),
                        "chow_p": (
                            chow["pvalue"]
                        ),
                        "hac_wald_stat": (
                            robust["statistic"]
                        ),
                        "hac_wald_p": (
                            robust["pvalue"]
                        ),
                        "beta_pre": (
                            robust["beta_pre"]
                        ),
                        "beta_post": (
                            robust["beta_post"]
                        ),
                        "beta_shift": (
                            robust["beta_shift"]
                        ),
                        "intercept_pre": (
                            robust[
                                "intercept_pre"
                            ]
                        ),
                        "intercept_post": (
                            robust[
                                "intercept_post"
                            ]
                        ),
                    }
                )

            unknown = detect_unknown_breaks(
                data=data,
                max_breaks=max_breaks,
                min_segment=min_segment,
            )

            unknown_details[
                f"{model_name}:{stage}"
            ] = unknown

            unknown_rows.append(
                {
                    "model": model_name,
                    "stage": stage,
                    "n": unknown["n"],
                    "selected_breaks": (
                        unknown[
                            "selected_breaks"
                        ]
                    ),
                    "selected_bic": (
                        unknown[
                            "selected_bic"
                        ]
                    ),
                    "break_months": (
                        "|".join(
                            unknown[
                                "break_months"
                            ]
                        )
                    ),
                }
            )

    predefined = pd.DataFrame(
        predefined_rows
    )

    reject, adjusted, _, _ = (
        multipletests(
            predefined[
                "hac_wald_p"
            ].to_numpy(),
            alpha=float(
                spec[
                    "multiplicity"
                ]["alpha"]
            ),
            method="holm",
        )
    )

    predefined[
        "hac_wald_p_holm"
    ] = adjusted

    predefined[
        "hac_wald_reject_holm"
    ] = reject

    unknown_table = pd.DataFrame(
        unknown_rows
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    predefined_path = (
        TABLE_ROOT
        / "structural_breaks_predefined.csv"
    )

    unknown_path = (
        TABLE_ROOT
        / "structural_breaks_unknown.csv"
    )

    predefined.to_csv(
        predefined_path,
        index=False,
    )

    unknown_table.to_csv(
        unknown_path,
        index=False,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        LOG_ROOT
        / "structural_stability.json"
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
        "predefined_breaks": (
            predefined_rows
        ),
        "unknown_breaks": (
            unknown_details
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
        "upstream_descriptive_manifest": (
            str(
                descriptive_manifest_path
            )
        ),
        "upstream_descriptive_manifest_sha256": (
            sha256_file(
                descriptive_manifest_path
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
        "predefined_table": (
            str(
                predefined_path
            )
        ),
        "predefined_table_sha256": (
            sha256_file(
                predefined_path
            )
        ),
        "unknown_table": (
            str(
                unknown_path
            )
        ),
        "unknown_table_sha256": (
            sha256_file(
                unknown_path
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
            "structural_stability_"
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
        "===== PREDEFINED BREAKS ====="
    )

    for _, row in predefined.iterrows():
        print(
            row["model"],
            "|",
            row["stage"],
            "|",
            row["break_month"],
            "| Chow p=",
            f"{row['chow_p']:.6g}",
            "| HAC Wald p=",
            f"{row['hac_wald_p']:.6g}",
            "| Holm=",
            f"{row['hac_wald_p_holm']:.6g}",
            "| beta",
            f"{row['beta_pre']:.4f}",
            "->",
            f"{row['beta_post']:.4f}",
            "| reject=",
            bool(
                row[
                    "hac_wald_reject_holm"
                ]
            ),
        )

    print()
    print(
        "===== UNKNOWN BREAKS / BIC ====="
    )

    for _, row in unknown_table.iterrows():
        breaks = (
            row["break_months"]
            if row["break_months"]
            else "none"
        )

        print(
            row["model"],
            "|",
            row["stage"],
            "| selected breaks=",
            int(
                row["selected_breaks"]
            ),
            "| months=",
            breaks,
            "| BIC=",
            f"{row['selected_bic']:.4f}",
        )

    print()
    print(
        f"Predefined: {predefined_path}"
    )
    print(
        f"Unknown: {unknown_path}"
    )
    print(
        f"Details: {details_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
