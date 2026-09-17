from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path("configs/research.yaml")
MANIFEST_ROOT = Path("manifests")
TABLE_ROOT = Path("outputs/tables")


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


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


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


def main() -> None:
    args = parse_args()

    cfg = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    spec = cfg["gold_scenario"]

    external_manifest_path = (
        MANIFEST_ROOT
        / (
            "external_anchors_"
            f"{args.external_run_id}.json"
        )
    )

    external_manifest = load_json(
        external_manifest_path
    )

    if (
        external_manifest.get("status")
        != "PASS"
    ):
        fail(
            "External anchor manifest not PASS"
        )

    gold_meta = external_manifest[
        "outputs"
    ]["gold_snapshot"]

    gold_path = Path(
        gold_meta["path"]
    )

    if (
        sha256_file(gold_path)
        != gold_meta["sha256"]
    ):
        fail(
            "Gold snapshot SHA-256 mismatch"
        )

    gold = load_json(
        gold_path
    )

    observation_date = spec[
        "observation_date"
    ]

    if (
        gold["observation_date"]
        != observation_date
    ):
        fail(
            "Gold observation date mismatch"
        )

    if (
        gold[
            "btc_supply_alignment_date"
        ]
        != observation_date
    ):
        fail(
            "Gold/BTC alignment date mismatch"
        )

    gold_value_trillion = float(
        gold[
            "reported_market_value_usd_trillion"
        ]
    )

    expected_gold_value = float(
        spec[
            "gold_input"
        ][
            "expected_market_value_usd_trillion"
        ]
    )

    if (
        gold_value_trillion
        != expected_gold_value
    ):
        fail(
            "Gold market value mismatch"
        )

    gold_value_usd = (
        gold_value_trillion
        * 1_000_000_000_000.0
    )

    monthly_manifest_path = (
        MANIFEST_ROOT
        / (
            "monthly_panel_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
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

    panel = pd.read_parquet(
        panel_path
    )

    panel[
        "period_end_utc"
    ] = pd.to_datetime(
        panel[
            "period_end_utc"
        ],
        utc=True,
    )

    row = panel[
        panel[
            "period_end_utc"
        ].dt.strftime(
            "%Y-%m-%d"
        )
        == observation_date
    ]

    if len(row) != 1:
        fail(
            "Expected exactly one aligned "
            "BTC observation"
        )

    row = row.iloc[0]

    btc_price = float(
        row[
            spec[
                "bitcoin_input"
            ]["price_field"]
        ]
    )

    btc_supply = float(
        row[
            spec[
                "bitcoin_input"
            ]["supply_field"]
        ]
    )

    if (
        btc_price <= 0
        or btc_supply <= 0
    ):
        fail(
            "BTC price and supply must be positive"
        )

    observed_btc_market_cap = (
        btc_price
        * btc_supply
    )

    spot_gold_share = (
        observed_btc_market_cap
        / gold_value_usd
    )

    rows = []

    for theta in spec[
        "scenarios"
    ]["gold_market_share"]:
        theta = float(theta)

        implied_market_cap = (
            theta
            * gold_value_usd
        )

        implied_price = (
            implied_market_cap
            / btc_supply
        )

        multiple = (
            implied_price
            / btc_price
        )

        log_gap = math.log(
            multiple
        )

        rows.append(
            {
                "observation_date": (
                    observation_date
                ),
                "gold_market_share": (
                    theta
                ),
                "gold_market_value_usd": (
                    gold_value_usd
                ),
                "implied_btc_market_cap_usd": (
                    implied_market_cap
                ),
                "btc_supply": (
                    btc_supply
                ),
                "observed_btc_price_usd": (
                    btc_price
                ),
                "observed_btc_market_cap_usd": (
                    observed_btc_market_cap
                ),
                "spot_gold_share": (
                    spot_gold_share
                ),
                "implied_btc_price_usd": (
                    implied_price
                ),
                "price_multiple": (
                    multiple
                ),
                "upside_vs_spot": (
                    multiple - 1.0
                ),
                "log_gap": (
                    log_gap
                ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_ROOT
        / "gold_market_share_scenarios.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "interpretation": (
            "scenario_sensitivity_only"
        ),
        "observation_date": (
            observation_date
        ),
        "gold_provider": (
            gold["provider"]
        ),
        "gold_market_value_usd": (
            gold_value_usd
        ),
        "gold_market_value_precision": (
            gold[
                "market_value_precision"
            ]
        ),
        "btc_price_usd": (
            btc_price
        ),
        "btc_supply": (
            btc_supply
        ),
        "observed_btc_market_cap_usd": (
            observed_btc_market_cap
        ),
        "spot_gold_share": (
            spot_gold_share
        ),
        "gold_snapshot": (
            str(gold_path)
        ),
        "gold_snapshot_sha256": (
            sha256_file(gold_path)
        ),
        "monthly_panel": (
            str(panel_path)
        ),
        "monthly_panel_sha256": (
            sha256_file(panel_path)
        ),
        "research_config_sha256": (
            sha256_file(CONFIG_PATH)
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
    }

    manifest_path = (
        MANIFEST_ROOT
        / (
            "gold_scenario_"
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
        "===== GOLD MARKET-SHARE SCENARIOS ====="
    )

    print(
        "Observation date:",
        observation_date,
    )

    print(
        "Gold market value:",
        f"${gold_value_usd / 1e12:.2f}T",
    )

    print(
        "BTC supply:",
        f"{btc_supply:,.2f}",
    )

    print(
        "BTC spot:",
        f"${btc_price:,.2f}",
    )

    print(
        "Observed BTC market cap:",
        f"${observed_btc_market_cap / 1e12:.4f}T",
    )

    print(
        "Observed BTC / gold share:",
        f"{spot_gold_share:.4%}",
    )

    for _, r in results.iterrows():
        print()

        print(
            f"{r['gold_market_share']:.0%}",
            "of gold market value",
        )

        print(
            "  implied market cap:",
            f"${r['implied_btc_market_cap_usd'] / 1e12:.2f}T",
        )

        print(
            "  implied BTC price:",
            f"${r['implied_btc_price_usd']:,.2f}",
        )

        print(
            "  price multiple:",
            f"{r['price_multiple']:.3f}x",
        )

        print(
            "  log gap:",
            f"{r['log_gap']:.6f}",
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
