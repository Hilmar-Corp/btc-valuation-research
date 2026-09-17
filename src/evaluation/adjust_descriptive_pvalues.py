from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


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

    manifest_path = (
        MANIFEST_ROOT
        / (
            "descriptive_relationships_"
            f"{args.base_run_id}_"
            f"{args.external_run_id}.json"
        )
    )

    if not manifest_path.exists():
        fail(
            f"Missing manifest: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("status") != "PASS":
        fail(
            "Descriptive relationship manifest "
            "is not PASS"
        )

    table_path = Path(
        manifest["summary_table"]
    )

    if (
        sha256_file(table_path)
        != manifest[
            "summary_table_sha256"
        ]
    ):
        fail(
            "Summary table SHA-256 mismatch"
        )

    df = pd.read_csv(
        table_path
    )

    required = {
        "model",
        "level_beta_p_hac",
        "diff_beta_p_hac",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        fail(
            f"Missing columns: {sorted(missing)}"
        )

    for stage, column in [
        (
            "level",
            "level_beta_p_hac",
        ),
        (
            "first_difference",
            "diff_beta_p_hac",
        ),
    ]:
        reject, adjusted, _, _ = (
            multipletests(
                df[column].to_numpy(),
                alpha=0.05,
                method="holm",
            )
        )

        df[
            f"{stage}_p_holm"
        ] = adjusted

        df[
            f"{stage}_reject_5pct_holm"
        ] = reject

    TABLE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        TABLE_ROOT
        / "descriptive_relationships_holm.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        "===== HOLM-ADJUSTED DESCRIPTIVE TESTS ====="
    )

    for _, row in df.iterrows():
        print()
        print(row["model"])

        print(
            "  level:",
            "raw=",
            row["level_beta_p_hac"],
            "| holm=",
            row["level_p_holm"],
            "| reject=",
            bool(
                row[
                    "level_reject_5pct_holm"
                ]
            ),
        )

        print(
            "  diff: ",
            "raw=",
            row["diff_beta_p_hac"],
            "| holm=",
            row[
                "first_difference_p_holm"
            ],
            "| reject=",
            bool(
                row[
                    "first_difference_"
                    "reject_5pct_holm"
                ]
            ),
        )

    print()
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
