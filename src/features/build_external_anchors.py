from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


CONTRACT_PATH = Path("configs/data_contract.yaml")
MANIFEST_ROOT = Path("manifests")
INTERIM_ROOT = Path("data/interim")
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


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


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


def get_contract_series(
    contract: dict[str, Any],
    series_id: str,
) -> dict[str, Any]:
    for item in contract["series"]:
        if item["id"] == series_id:
            return item

    fail(
        f"Contract series not found: {series_id}"
    )


def get_manifest_entry(
    manifest: dict[str, Any],
    series_id: str,
) -> dict[str, Any]:
    matches = [
        x
        for x in manifest["files"]
        if x["series_id"] == series_id
    ]

    if len(matches) != 1:
        fail(
            f"Expected exactly one manifest entry "
            f"for {series_id}; found {len(matches)}"
        )

    return matches[0]


def verify_manifest_file(
    entry: dict[str, Any],
) -> Path:
    path = Path(entry["path"])

    if not path.exists():
        fail(
            f"Missing raw source: {path}"
        )

    actual = sha256_file(path)
    expected = entry["sha256"]

    if actual != expected:
        fail(
            f"SHA-256 mismatch: {path}"
        )

    return path


def make_daily_index(
    start: str,
    end: str,
) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(
            start,
            tz="UTC",
        ),
        end=pd.Timestamp(
            end,
            tz="UTC",
        ),
        freq="D",
    )


def assert_exact_index(
    series_name: str,
    actual: pd.DatetimeIndex,
    expected: pd.DatetimeIndex,
) -> None:
    if actual.equals(expected):
        return

    missing = expected.difference(actual)
    extra = actual.difference(expected)

    fail(
        f"{series_name}: calendar mismatch | "
        f"actual={len(actual)} | "
        f"expected={len(expected)} | "
        f"missing={len(missing)} | "
        f"extra={len(extra)}"
    )


def load_coinmetrics_metric(
    entry: dict[str, Any],
    metric: str,
) -> pd.DataFrame:
    path = verify_manifest_file(entry)
    payload = load_json(path)

    data = payload.get("data")

    if not isinstance(data, list):
        fail(
            f"{metric}: raw response has no data list"
        )

    rows = []

    for item in data:
        if metric not in item:
            fail(
                f"{metric}: field absent from raw row"
            )

        rows.append(
            {
                "time": item.get("time"),
                metric: item.get(metric),
            }
        )

    df = pd.DataFrame(rows)

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce",
    )

    if df["time"].isna().any():
        fail(
            f"{metric}: invalid timestamp"
        )

    if df[metric].isna().any():
        fail(
            f"{metric}: non-numeric/null value"
        )

    if not np.isfinite(
        df[metric].to_numpy(
            dtype=float
        )
    ).all():
        fail(
            f"{metric}: non-finite value"
        )

    df = (
        df.sort_values("time")
        .reset_index(drop=True)
    )

    duplicates = int(
        df["time"].duplicated().sum()
    )

    if duplicates:
        fail(
            f"{metric}: "
            f"{duplicates} duplicate timestamps"
        )

    return df


def load_cbeci(
    entry: dict[str, Any],
    native_field: str,
) -> pd.DataFrame:
    path = verify_manifest_file(entry)

    raw = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = raw.splitlines()

    if len(lines) < 3:
        fail(
            "CBECI source is unexpectedly short"
        )

    if (
        "Average electricity cost assumption"
        not in lines[0]
    ):
        fail(
            "CBECI metadata header not found"
        )

    if "0.05" not in lines[0]:
        fail(
            "CBECI electricity-price assumption "
            "is not 0.05 USD/kWh"
        )

    df = pd.read_csv(
        io.StringIO(raw),
        skiprows=1,
    )

    required = {
        "Timestamp",
        "Date and Time",
        native_field,
    }

    missing_columns = (
        required - set(df.columns)
    )

    if missing_columns:
        fail(
            "CBECI missing columns: "
            f"{sorted(missing_columns)}"
        )

    out = df[
        [
            "Date and Time",
            native_field,
        ]
    ].copy()

    out.columns = [
        "time",
        "cbeci_annualised_consumption_twh",
    ]

    out["time"] = pd.to_datetime(
        out["time"],
        utc=True,
        errors="coerce",
    )

    out[
        "cbeci_annualised_consumption_twh"
    ] = pd.to_numeric(
        out[
            "cbeci_annualised_consumption_twh"
        ],
        errors="coerce",
    )

    if out["time"].isna().any():
        fail(
            "CBECI contains invalid timestamps"
        )

    out = (
        out.sort_values("time")
        .reset_index(drop=True)
    )

    duplicates = int(
        out["time"].duplicated().sum()
    )

    if duplicates:
        fail(
            f"CBECI contains "
            f"{duplicates} duplicate timestamps"
        )

    return out


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


def save_json(
    payload: dict[str, Any],
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

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()

    base_manifest_path = (
        MANIFEST_ROOT
        / f"normalized_{args.base_run_id}.json"
    )

    external_manifest_path = (
        MANIFEST_ROOT
        / (
            f"external_download_"
            f"{args.external_run_id}.json"
        )
    )

    if not base_manifest_path.exists():
        fail(
            f"Missing certified base manifest: "
            f"{base_manifest_path}"
        )

    if not external_manifest_path.exists():
        fail(
            f"Missing external manifest: "
            f"{external_manifest_path}"
        )

    base_manifest = load_json(
        base_manifest_path
    )

    external_manifest = load_json(
        external_manifest_path
    )

    if base_manifest.get("status") != "PASS":
        fail(
            "Base normalized manifest "
            "does not have PASS status"
        )

    if external_manifest.get("status") != "PASS":
        fail(
            "External manifest "
            "does not have PASS status"
        )

    if (
        external_manifest[
            "base_certified_run_id"
        ]
        != args.base_run_id
    ):
        fail(
            "External manifest refers to "
            "a different base run"
        )

    contract = load_contract()

    production_spec = get_contract_series(
        contract,
        "btc_production_cost_electricity",
    )

    cbeci_spec = get_contract_series(
        contract,
        "cbeci_annualised_consumption_guess",
    )

    gold_spec = get_contract_series(
        contract,
        "gold_market_value_snapshot",
    )

    construction = production_spec[
        "construction"
    ]

    electricity_price = float(
        construction[
            "electricity_price_usd_per_kwh"
        ]
    )

    annualisation_days = float(
        construction[
            "annualisation_days"
        ]
    )

    reward_window = int(
        construction[
            "reward_smoothing_days"
        ]
    )

    warmup_days = int(
        construction[
            "reward_warmup_days"
        ]
    )

    if reward_window != 7:
        fail(
            "Expected frozen reward window of 7 days"
        )

    if warmup_days != reward_window - 1:
        fail(
            "Reward warm-up does not equal "
            "window minus one"
        )

    sample_start = external_manifest[
        "sample_start"
    ]

    sample_end = external_manifest[
        "sample_end_inclusive"
    ]

    reward_start = external_manifest[
        "reward_input_start"
    ]

    expected_sample = make_daily_index(
        sample_start,
        sample_end,
    )

    expected_rewards = make_daily_index(
        reward_start,
        sample_end,
    )

    print(
        "===== BUILD EXTERNAL ANCHORS ====="
    )

    print(
        f"Base run: {args.base_run_id}"
    )

    print(
        f"External run: "
        f"{args.external_run_id}"
    )

    print(
        f"Sample days: "
        f"{len(expected_sample)}"
    )

    print(
        f"Reward-input days: "
        f"{len(expected_rewards)}"
    )

    issuance_entry = get_manifest_entry(
        external_manifest,
        "btc_issuance_native",
    )

    fees_entry = get_manifest_entry(
        external_manifest,
        "btc_fees_native",
    )

    cbeci_entry = get_manifest_entry(
        external_manifest,
        "cbeci_annualised_consumption_guess",
    )

    gold_entry = get_manifest_entry(
        external_manifest,
        "gold_market_value_snapshot",
    )

    issuance = load_coinmetrics_metric(
        issuance_entry,
        "IssTotNtv",
    )

    fees = load_coinmetrics_metric(
        fees_entry,
        "FeeTotNtv",
    )

    assert_exact_index(
        "IssTotNtv",
        pd.DatetimeIndex(
            issuance["time"]
        ),
        expected_rewards,
    )

    assert_exact_index(
        "FeeTotNtv",
        pd.DatetimeIndex(
            fees["time"]
        ),
        expected_rewards,
    )

    if (
        issuance["IssTotNtv"] < 0
    ).any():
        fail(
            "IssTotNtv contains negative values"
        )

    if (
        fees["FeeTotNtv"] < 0
    ).any():
        fail(
            "FeeTotNtv contains negative values"
        )

    rewards = issuance.merge(
        fees,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    rewards["miner_reward_btc"] = (
        rewards["IssTotNtv"]
        + rewards["FeeTotNtv"]
    )

    if (
        rewards["miner_reward_btc"] <= 0
    ).any():
        fail(
            "Total miner reward is non-positive"
        )

    rewards[
        "miner_reward_ma7_btc"
    ] = (
        rewards[
            "miner_reward_btc"
        ]
        .rolling(
            window=reward_window,
            min_periods=reward_window,
        )
        .mean()
    )

    rewards_sample = rewards.loc[
        rewards["time"].between(
            pd.Timestamp(
                sample_start,
                tz="UTC",
            ),
            pd.Timestamp(
                sample_end,
                tz="UTC",
            ),
        )
    ].copy()

    rewards_sample = (
        rewards_sample
        .reset_index(drop=True)
    )

    assert_exact_index(
        "Smoothed miner rewards",
        pd.DatetimeIndex(
            rewards_sample["time"]
        ),
        expected_sample,
    )

    if rewards_sample[
        "miner_reward_ma7_btc"
    ].isna().any():
        fail(
            "Sample contains incomplete "
            "reward rolling windows"
        )

    cbeci = load_cbeci(
        cbeci_entry,
        cbeci_spec["native_field"],
    )

    cbeci_sample = cbeci.loc[
        cbeci["time"].between(
            pd.Timestamp(
                sample_start,
                tz="UTC",
            ),
            pd.Timestamp(
                sample_end,
                tz="UTC",
            ),
        )
    ].copy()

    cbeci_sample = (
        cbeci_sample
        .reset_index(drop=True)
    )

    assert_exact_index(
        "CBECI",
        pd.DatetimeIndex(
            cbeci_sample["time"]
        ),
        expected_sample,
    )

    field = (
        "cbeci_annualised_consumption_twh"
    )

    if cbeci_sample[field].isna().any():
        fail(
            "CBECI sample contains NaN"
        )

    if not np.isfinite(
        cbeci_sample[field].to_numpy(
            dtype=float
        )
    ).all():
        fail(
            "CBECI sample contains "
            "non-finite values"
        )

    if (
        cbeci_sample[field] <= 0
    ).any():
        fail(
            "CBECI sample contains "
            "non-positive consumption"
        )

    anchor = cbeci_sample.merge(
        rewards_sample,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    if len(anchor) != len(expected_sample):
        fail(
            "Production-cost anchor has "
            "unexpected row count"
        )

    anchor[
        "electricity_price_usd_per_kwh"
    ] = electricity_price

    anchor[
        "annualisation_days"
    ] = annualisation_days

    anchor[
        "electricity_spend_usd_per_day"
    ] = (
        anchor[
            "cbeci_annualised_consumption_twh"
        ]
        * 1e9
        * electricity_price
        / annualisation_days
    )

    anchor[
        "production_cost_electricity_usd_per_btc"
    ] = (
        anchor[
            "electricity_spend_usd_per_day"
        ]
        / anchor[
            "miner_reward_ma7_btc"
        ]
    )

    numeric_checks = [
        "electricity_spend_usd_per_day",
        "production_cost_electricity_usd_per_btc",
    ]

    for column in numeric_checks:
        values = anchor[
            column
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            fail(
                f"{column} contains "
                "non-finite values"
            )

        if (
            anchor[column] <= 0
        ).any():
            fail(
                f"{column} contains "
                "non-positive values"
            )

    anchor = anchor[
        [
            "time",
            "cbeci_annualised_consumption_twh",
            "IssTotNtv",
            "FeeTotNtv",
            "miner_reward_btc",
            "miner_reward_ma7_btc",
            "electricity_price_usd_per_kwh",
            "annualisation_days",
            "electricity_spend_usd_per_day",
            "production_cost_electricity_usd_per_btc",
        ]
    ].copy()

    assert_exact_index(
        "Final production-cost anchor",
        pd.DatetimeIndex(
            anchor["time"]
        ),
        expected_sample,
    )

    gold_path = verify_manifest_file(
        gold_entry
    )

    evidence = gold_entry.get(
        "evidence_checks",
        {}
    )

    if not all(
        evidence.get(key) is True
        for key in (
            "contains_222600",
            "contains_29tn",
            "contains_q2_2026",
        )
    ):
        fail(
            "WGC HTML evidence checks "
            "did not all pass"
        )

    gold_snapshot = {
        "series_id": (
            "gold_market_value_snapshot"
        ),
        "provider": (
            "World Gold Council"
        ),
        "observation_date": (
            gold_spec["observation_date"]
        ),
        "publication_date": (
            gold_spec["publication_date"]
        ),
        "above_ground_stock_tonnes": (
            gold_spec[
                "above_ground_stock_tonnes"
            ]
        ),
        "reported_market_value_usd_trillion": (
            gold_spec[
                "reported_market_value_usd_trillion"
            ]
        ),
        "market_value_precision": (
            "reported_rounded_snapshot"
        ),
        "btc_supply_alignment_date": (
            gold_spec["observation_date"]
        ),
        "scenario_use_only": True,
        "source_raw_path": str(
            gold_path
        ),
        "source_raw_sha256": (
            sha256_file(gold_path)
        ),
        "evidence_checks": evidence,
    }

    output_dir = (
        INTERIM_ROOT
        / args.base_run_id
        / f"external_{args.external_run_id}"
    )

    if output_dir.exists():
        fail(
            f"Output directory already exists: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    production_meta = save_parquet(
        anchor,
        output_dir
        / "production_cost_electricity.parquet",
    )

    gold_meta = save_json(
        gold_snapshot,
        output_dir
        / "gold_market_snapshot.json",
    )

    qc_report = {
        "schema_version": "1.0.0",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "base_run_id": args.base_run_id,
        "external_run_id": (
            args.external_run_id
        ),
        "sample": {
            "rows": int(
                len(anchor)
            ),
            "first": (
                anchor["time"]
                .min()
                .isoformat()
            ),
            "last": (
                anchor["time"]
                .max()
                .isoformat()
            ),
            "missing_cells": int(
                anchor
                .isna()
                .sum()
                .sum()
            ),
        },
        "reward_inputs": {
            "warmup_rows": (
                warmup_days
            ),
            "total_raw_rows": int(
                len(rewards)
            ),
            "sample_rows": int(
                len(rewards_sample)
            ),
            "rolling_window_days": (
                reward_window
            ),
            "partial_windows_in_sample": int(
                rewards_sample[
                    "miner_reward_ma7_btc"
                ]
                .isna()
                .sum()
            ),
        },
        "cbeci": {
            "sample_rows": int(
                len(cbeci_sample)
            ),
            "minimum_twh": float(
                cbeci_sample[field].min()
            ),
            "maximum_twh": float(
                cbeci_sample[field].max()
            ),
        },
        "production_cost": {
            "electricity_price_usd_per_kwh": (
                electricity_price
            ),
            "annualisation_days": (
                annualisation_days
            ),
            "minimum_usd_per_btc": float(
                anchor[
                    "production_cost_electricity_usd_per_btc"
                ].min()
            ),
            "maximum_usd_per_btc": float(
                anchor[
                    "production_cost_electricity_usd_per_btc"
                ].max()
            ),
            "first_usd_per_btc": float(
                anchor[
                    "production_cost_electricity_usd_per_btc"
                ].iloc[0]
            ),
            "last_usd_per_btc": float(
                anchor[
                    "production_cost_electricity_usd_per_btc"
                ].iloc[-1]
            ),
        },
        "gold_snapshot": {
            "observation_date": (
                gold_snapshot[
                    "observation_date"
                ]
            ),
            "reported_market_value_usd_trillion": (
                gold_snapshot[
                    "reported_market_value_usd_trillion"
                ]
            ),
            "evidence_checks": (
                evidence
            ),
        },
        "environment": {
            "python": (
                platform.python_version()
            ),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": (
                platform.platform()
            ),
        },
        "status": "PASS",
    }

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    qc_path = (
        LOG_ROOT
        / (
            f"external_anchors_qc_"
            f"{args.external_run_id}.json"
        )
    )

    qc_path.write_text(
        json.dumps(
            qc_report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "base_run_id": args.base_run_id,
        "external_run_id": (
            args.external_run_id
        ),
        "contract_version": (
            contract["contract"]["version"]
        ),
        "contract_sha256": (
            sha256_file(
                CONTRACT_PATH
            )
        ),
        "source_external_manifest": (
            str(
                external_manifest_path
            )
        ),
        "source_external_manifest_sha256": (
            sha256_file(
                external_manifest_path
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
        "qc_report": str(
            qc_path
        ),
        "qc_report_sha256": (
            sha256_file(
                qc_path
            )
        ),
        "outputs": {
            "production_cost": (
                production_meta
            ),
            "gold_snapshot": (
                gold_meta
            ),
        },
    }

    manifest_path = (
        MANIFEST_ROOT
        / (
            f"external_anchors_"
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
            output_manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "[OK] Issuance + fees aligned "
        f"| rows={len(rewards)}"
    )

    print(
        "[OK] MA7 rewards complete in sample "
        f"| rows={len(rewards_sample)}"
    )

    print(
        "[OK] CBECI aligned "
        f"| rows={len(cbeci_sample)}"
    )

    print(
        "[OK] Production-cost series built "
        f"| rows={len(anchor)}"
    )

    print(
        "[OK] WGC snapshot normalized"
    )

    print()
    print(
        "===== EXTERNAL ANCHORS CERTIFIED ====="
    )

    print(
        f"Production cost first: "
        f"${anchor['production_cost_electricity_usd_per_btc'].iloc[0]:,.2f}"
    )

    print(
        f"Production cost last: "
        f"${anchor['production_cost_electricity_usd_per_btc'].iloc[-1]:,.2f}"
    )

    print(
        f"Production cost min: "
        f"${anchor['production_cost_electricity_usd_per_btc'].min():,.2f}"
    )

    print(
        f"Production cost max: "
        f"${anchor['production_cost_electricity_usd_per_btc'].max():,.2f}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
