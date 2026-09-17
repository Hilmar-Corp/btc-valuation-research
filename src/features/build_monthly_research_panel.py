from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


MANIFEST_ROOT = Path("manifests")
PROCESSED_ROOT = Path("data/processed")
LOG_ROOT = Path("outputs/logs")
RESEARCH_CONFIG_PATH = Path("configs/research.yaml")


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


def verify_file(
    path: Path,
    expected_sha256: str,
) -> None:
    if not path.exists():
        fail(
            f"Missing input file: {path}"
        )

    actual = sha256_file(path)

    if actual != expected_sha256:
        fail(
            f"SHA-256 mismatch: {path}"
        )


def get_daily_panel_meta(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["outputs"]
        if Path(item["path"]).name
        == "panel_daily.parquet"
    ]

    if len(matches) != 1:
        fail(
            "Expected exactly one "
            "panel_daily.parquet"
        )

    return matches[0]


def normalize_daily(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    df = df.copy()

    if "time" not in df.columns:
        fail(
            f"{label}: missing time column"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    if df["time"].isna().any():
        fail(
            f"{label}: invalid timestamp"
        )

    df = (
        df.sort_values("time")
        .reset_index(drop=True)
    )

    if df["time"].duplicated().any():
        fail(
            f"{label}: duplicate timestamps"
        )

    if not df["time"].is_monotonic_increasing:
        fail(
            f"{label}: non-monotonic timestamps"
        )

    return df


def complete_month(
    group: pd.DataFrame,
) -> tuple[bool, int]:
    first = group["time"].iloc[0]
    last = group["time"].iloc[-1]

    year = int(first.year)
    month = int(first.month)

    expected_days = calendar.monthrange(
        year,
        month,
    )[1]

    expected_first = pd.Timestamp(
        year=year,
        month=month,
        day=1,
        tz="UTC",
    )

    expected_last = pd.Timestamp(
        year=year,
        month=month,
        day=expected_days,
        tz="UTC",
    )

    ok = (
        len(group) == expected_days
        and first == expected_first
        and last == expected_last
    )

    return ok, expected_days


def save_parquet(
    df: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    if path.exists():
        fail(
            f"Refusing to overwrite: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        path,
        index=False,
    )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
    }


def main() -> None:
    args = parse_args()

    config = load_yaml(
        RESEARCH_CONFIG_PATH
    )

    expected_count = int(
        config["sample"][
            "expected_primary_months"
        ]
    )

    expected_months = [
        str(x)
        for x in pd.period_range(
            start=config["sample"][
                "primary_monthly_start"
            ],
            end=config["sample"][
                "primary_monthly_end"
            ],
            freq="M",
        )
    ]

    if len(expected_months) != expected_count:
        fail(
            "Configured monthly range does not "
            "match expected_primary_months"
        )

    if (
        config["sample_policy"][
            "primary_monthly_sample"
        ]
        != "complete_calendar_months_only"
    ):
        fail(
            "Unexpected monthly sample policy"
        )

    if (
        config["sample_policy"][
            "partial_months"
        ]
        != "exclude"
    ):
        fail(
            "Partial months are not frozen "
            "as excluded"
        )

    if (
        config["model_scope"][
            "adoption"
        ]["enabled"]
        is not False
    ):
        fail(
            "Adoption model unexpectedly enabled"
        )

    base_manifest_path = (
        MANIFEST_ROOT
        / f"normalized_{args.base_run_id}.json"
    )

    anchor_manifest_path = (
        MANIFEST_ROOT
        / (
            f"external_anchors_"
            f"{args.external_run_id}.json"
        )
    )

    if not base_manifest_path.exists():
        fail(
            f"Missing manifest: "
            f"{base_manifest_path}"
        )

    if not anchor_manifest_path.exists():
        fail(
            f"Missing manifest: "
            f"{anchor_manifest_path}"
        )

    base_manifest = load_json(
        base_manifest_path
    )

    anchor_manifest = load_json(
        anchor_manifest_path
    )

    if base_manifest.get("status") != "PASS":
        fail(
            "Base normalized dataset "
            "is not PASS"
        )

    if anchor_manifest.get("status") != "PASS":
        fail(
            "External anchors are not PASS"
        )

    if (
        anchor_manifest["base_run_id"]
        != args.base_run_id
    ):
        fail(
            "External anchors refer to "
            "another base run"
        )

    if (
        anchor_manifest["external_run_id"]
        != args.external_run_id
    ):
        fail(
            "External run-id mismatch"
        )

    daily_meta = get_daily_panel_meta(
        base_manifest
    )

    daily_path = Path(
        daily_meta["path"]
    )

    verify_file(
        daily_path,
        daily_meta["sha256"],
    )

    cost_meta = (
        anchor_manifest[
            "outputs"
        ]["production_cost"]
    )

    cost_path = Path(
        cost_meta["path"]
    )

    verify_file(
        cost_path,
        cost_meta["sha256"],
    )

    daily = normalize_daily(
        pd.read_parquet(
            daily_path
        ),
        "Daily base panel",
    )

    cost = normalize_daily(
        pd.read_parquet(
            cost_path
        ),
        "Production-cost panel",
    )

    required = {
        "time",
        "btc_price_usd",
        "btc_supply_current",
        "btc_active_addresses",
        "btc_hashrate",
    }

    missing = required - set(
        daily.columns
    )

    if missing:
        fail(
            f"Daily panel missing: "
            f"{sorted(missing)}"
        )

    cost_col = (
        "production_cost_electricity_usd_per_btc"
    )

    if cost_col not in cost.columns:
        fail(
            "Production-cost variable missing"
        )

    daily_index = pd.DatetimeIndex(
        daily["time"]
    )

    cost_index = pd.DatetimeIndex(
        cost["time"]
    )

    if not daily_index.equals(
        cost_index
    ):
        fail(
            "Daily base and production-cost "
            "calendars differ"
        )

    if len(daily) != 3317:
        fail(
            f"Unexpected daily row count: "
            f"{len(daily)}"
        )

    day_diffs = (
        daily["time"]
        .diff()
        .dropna()
    )

    if not (
        day_diffs
        == pd.Timedelta(days=1)
    ).all():
        fail(
            "Daily panel is not a complete "
            "one-day calendar"
        )

    panel = daily.merge(
        cost[
            [
                "time",
                cost_col,
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )

    if len(panel) != 3317:
        fail(
            "Daily merge changed row count"
        )

    if panel.isna().any().any():
        fail(
            "Unexpected missing values "
            "before scarcity construction"
        )

    panel[
        "btc_supply_lag_365d"
    ] = (
        panel[
            "btc_supply_current"
        ]
        .shift(365)
    )

    panel[
        "btc_flow_365d"
    ] = (
        panel[
            "btc_supply_current"
        ]
        - panel[
            "btc_supply_lag_365d"
        ]
    )

    bad_flow = (
        panel[
            "btc_flow_365d"
        ].notna()
        &
        (
            panel[
                "btc_flow_365d"
            ]
            <= 0
        )
    )

    if bad_flow.any():
        fail(
            "Non-positive 365-day BTC flow"
        )

    panel[
        "btc_stock_to_flow"
    ] = (
        panel[
            "btc_supply_current"
        ]
        / panel[
            "btc_flow_365d"
        ]
    )

    panel["month"] = (
        panel["time"]
        .dt.strftime("%Y-%m")
    )

    rows = []
    excluded = []

    for month_id, group in panel.groupby(
        "month",
        sort=True,
    ):
        group = (
            group
            .sort_values("time")
            .reset_index(drop=True)
        )

        is_full, expected_days = (
            complete_month(group)
        )

        if not is_full:
            excluded.append(
                {
                    "month": month_id,
                    "observed_days": int(
                        len(group)
                    ),
                    "expected_days": int(
                        expected_days
                    ),
                    "reason": (
                        "incomplete_calendar_month"
                    ),
                }
            )

            continue

        last = group.iloc[-1]

        flow = last[
            "btc_flow_365d"
        ]

        stock_to_flow = last[
            "btc_stock_to_flow"
        ]

        rows.append(
            {
                "month": month_id,
                "period_end_utc": (
                    last["time"]
                ),
                "information_available_at_utc": (
                    last["time"]
                    + pd.Timedelta(days=1)
                ),
                "calendar_days": int(
                    expected_days
                ),
                "btc_price_usd": float(
                    last[
                        "btc_price_usd"
                    ]
                ),
                "btc_supply_current": float(
                    last[
                        "btc_supply_current"
                    ]
                ),
                "btc_active_addresses": float(
                    group[
                        "btc_active_addresses"
                    ].mean()
                ),
                "btc_hashrate": float(
                    group[
                        "btc_hashrate"
                    ].mean()
                ),
                "production_cost_electricity_usd_per_btc": float(
                    group[
                        cost_col
                    ].mean()
                ),
                "btc_flow_365d": (
                    float(flow)
                    if pd.notna(flow)
                    else np.nan
                ),
                "btc_stock_to_flow": (
                    float(stock_to_flow)
                    if pd.notna(
                        stock_to_flow
                    )
                    else np.nan
                ),
            }
        )

    monthly = pd.DataFrame(rows)

    if monthly["month"].tolist() != expected_months:
        actual = set(
            monthly["month"]
        )

        expected = set(
            expected_months
        )

        fail(
            "Monthly sample differs from "
            "frozen sample | "
            f"missing={sorted(expected - actual)} | "
            f"extra={sorted(actual - expected)}"
        )

    if len(monthly) != expected_count:
        fail(
            f"Monthly rows={len(monthly)} "
            f"!= {expected_count}"
        )

    if len(excluded) != 2:
        fail(
            f"Expected exactly 2 partial "
            f"months; found {len(excluded)}"
        )

    monthly[
        "btc_market_cap_usd"
    ] = (
        monthly[
            "btc_price_usd"
        ]
        * monthly[
            "btc_supply_current"
        ]
    )

    positive_columns = [
        "btc_price_usd",
        "btc_supply_current",
        "btc_active_addresses",
        "btc_hashrate",
        "production_cost_electricity_usd_per_btc",
        "btc_market_cap_usd",
    ]

    for column in positive_columns:
        values = monthly[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            fail(
                f"{column}: non-finite value"
            )

        if (
            monthly[column] <= 0
        ).any():
            fail(
                f"{column}: non-positive value"
            )

    monthly[
        "log_btc_price"
    ] = np.log(
        monthly[
            "btc_price_usd"
        ]
    )

    monthly[
        "log_btc_market_cap"
    ] = np.log(
        monthly[
            "btc_market_cap_usd"
        ]
    )

    monthly[
        "log_active_addresses"
    ] = np.log(
        monthly[
            "btc_active_addresses"
        ]
    )

    monthly[
        "log_production_cost_electricity"
    ] = np.log(
        monthly[
            "production_cost_electricity_usd_per_btc"
        ]
    )

    monthly[
        "scarcity_available"
    ] = (
        monthly[
            "btc_stock_to_flow"
        ].notna()
        &
        (
            monthly[
                "btc_stock_to_flow"
            ]
            > 0
        )
    )

    monthly[
        "log_stock_to_flow"
    ] = np.nan

    scarcity_mask = (
        monthly[
            "scarcity_available"
        ]
    )

    monthly.loc[
        scarcity_mask,
        "log_stock_to_flow",
    ] = np.log(
        monthly.loc[
            scarcity_mask,
            "btc_stock_to_flow",
        ]
    )

    scarcity_rows = int(
        monthly[
            "scarcity_available"
        ].sum()
    )

    scarcity_missing = int(
        (
            ~monthly[
                "scarcity_available"
            ]
        ).sum()
    )

    if scarcity_rows != 97:
        fail(
            f"Expected 97 scarcity rows; "
            f"found {scarcity_rows}"
        )

    if scarcity_missing != 11:
        fail(
            f"Expected 11 pre-scarcity rows; "
            f"found {scarcity_missing}"
        )

    first_scarcity_month = (
        monthly.loc[
            monthly[
                "scarcity_available"
            ],
            "month",
        ].iloc[0]
    )

    if first_scarcity_month != "2018-08":
        fail(
            "Unexpected first scarcity month: "
            f"{first_scarcity_month}"
        )

    scarcity_nan_columns = [
        "btc_flow_365d",
        "btc_stock_to_flow",
        "log_stock_to_flow",
    ]

    for column in scarcity_nan_columns:
        if int(
            monthly[
                column
            ].isna()
            .sum()
        ) != 11:
            fail(
                f"{column}: expected 11 NaN"
            )

    core_columns = [
        "month",
        "period_end_utc",
        "information_available_at_utc",
        "calendar_days",
        "btc_price_usd",
        "btc_supply_current",
        "btc_active_addresses",
        "btc_hashrate",
        "production_cost_electricity_usd_per_btc",
        "btc_market_cap_usd",
        "log_btc_price",
        "log_btc_market_cap",
        "log_active_addresses",
        "log_production_cost_electricity",
        "scarcity_available",
    ]

    if (
        monthly[
            core_columns
        ]
        .isna()
        .any()
        .any()
    ):
        fail(
            "Unexpected missing value "
            "in core monthly variables"
        )

    output_dir = (
        PROCESSED_ROOT
        / args.base_run_id
        / (
            f"external_"
            f"{args.external_run_id}"
        )
    )

    output_path = (
        output_dir
        / "monthly_research_panel.parquet"
    )

    output_meta = save_parquet(
        monthly,
        output_path,
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    qc_path = (
        LOG_ROOT
        / (
            f"monthly_panel_qc_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    qc_report = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "base_run_id": (
            args.base_run_id
        ),
        "external_run_id": (
            args.external_run_id
        ),
        "primary_months": int(
            len(monthly)
        ),
        "first_month": (
            monthly[
                "month"
            ].iloc[0]
        ),
        "last_month": (
            monthly[
                "month"
            ].iloc[-1]
        ),
        "excluded_partial_months": (
            excluded
        ),
        "network_rows": int(
            len(monthly)
        ),
        "production_cost_rows": int(
            len(monthly)
        ),
        "scarcity_rows": (
            scarcity_rows
        ),
        "scarcity_missing_rows": (
            scarcity_missing
        ),
        "first_scarcity_month": (
            first_scarcity_month
        ),
        "common_model_sample_rows": (
            scarcity_rows
        ),
        "adoption_enabled": False,
        "missingness": {
            column: int(
                monthly[
                    column
                ].isna()
                .sum()
            )
            for column in monthly.columns
        },
    }

    qc_path.write_text(
        json.dumps(
            qc_report,
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
        "research_config": (
            str(
                RESEARCH_CONFIG_PATH
            )
        ),
        "research_config_sha256": (
            sha256_file(
                RESEARCH_CONFIG_PATH
            )
        ),
        "base_manifest": (
            str(
                base_manifest_path
            )
        ),
        "base_manifest_sha256": (
            sha256_file(
                base_manifest_path
            )
        ),
        "external_anchor_manifest": (
            str(
                anchor_manifest_path
            )
        ),
        "external_anchor_manifest_sha256": (
            sha256_file(
                anchor_manifest_path
            )
        ),
        "builder_path": (
            str(Path(__file__))
        ),
        "builder_sha256": (
            sha256_file(
                Path(__file__)
            )
        ),
        "qc_report": (
            str(qc_path)
        ),
        "qc_report_sha256": (
            sha256_file(
                qc_path
            )
        ),
        "output": (
            output_meta
        ),
    }

    manifest_path = (
        MANIFEST_ROOT
        / (
            f"monthly_panel_"
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
        "===== MONTHLY RESEARCH PANEL ====="
    )

    print(
        f"[OK] Primary months: "
        f"{len(monthly)}"
    )

    print(
        f"[OK] First month: "
        f"{monthly['month'].iloc[0]}"
    )

    print(
        f"[OK] Last month: "
        f"{monthly['month'].iloc[-1]}"
    )

    print(
        f"[OK] Partial months excluded: "
        f"{len(excluded)}"
    )

    print(
        f"[OK] Network observations: "
        f"{len(monthly)}"
    )

    print(
        f"[OK] Production-cost observations: "
        f"{len(monthly)}"
    )

    print(
        f"[OK] Scarcity observations: "
        f"{scarcity_rows}"
    )

    print(
        f"[OK] First scarcity month: "
        f"{first_scarcity_month}"
    )

    print(
        f"[OK] Common model sample: "
        f"{scarcity_rows}"
    )

    print()
    print(
        f"Output: {output_path}"
    )

    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
