from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]


FORBIDDEN_TOP_LEVEL = {
    "data",
    "outputs",
    "manifests",
}


FORBIDDEN_EXACT_PATHS = {
    "configs/post_results_regression_lock.yaml",
}


FORBIDDEN_NAME_FRAGMENTS = {
    "final_evidence_pack",
    "oos_metrics",
    "oos_valuation",
    "oos_loss",
    "gap_predictive",
    "gap_shared_price",
    "stationarity_results",
    "structural_breaks",
    "descriptive_relationships",
    "gold_market_share_scenarios",
    "reproducibility_environment",
}


FORBIDDEN_DATA_SUFFIXES = {
    ".csv",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".xlsx",
    ".xls",
}


FORBIDDEN_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
}


SECRET_PATTERNS = [
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    re.compile(
        r"gh[pousr]_[A-Za-z0-9_]{30,}"
    ),
    re.compile(
        r"AKIA[0-9A-Z]{16}"
    ),
    re.compile(
        r"(?i)(?:api[_-]?key|secret|password|access[_-]?token)"
        r"\s*[:=]\s*[\"'][A-Za-z0-9/+_.=-]{16,}[\"']"
    ),
]


REQUIRED_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "PUBLICATION_SCOPE.md",
    "REPRODUCIBILITY.md",
    "SECURITY.md",
    "METHODOLOGY.md",
    "CHANGE_CONTROL.md",
    "CITATION.cff",
    "research_spec.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/workflows/quality.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/security.yml",
}


def fail(
    message: str,
) -> None:
    print(
        f"[FAIL] {message}",
        file=sys.stderr,
    )

    raise SystemExit(1)


def iter_files():
    ignored_directories = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "__pycache__",
    }

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in ignored_directories
            for part in path.parts
        ):
            continue

        yield path


def relative(
    path: Path,
) -> str:
    return str(
        path.relative_to(
            ROOT
        )
    )


def check_required_files() -> None:
    missing = [
        name
        for name in sorted(
            REQUIRED_FILES
        )
        if not (
            ROOT
            / name
        ).exists()
    ]

    if missing:
        fail(
            "missing required files: "
            + ", ".join(
                missing
            )
        )


def check_forbidden_paths() -> None:
    for top in (
        FORBIDDEN_TOP_LEVEL
    ):
        if (
            ROOT
            / top
        ).exists():
            fail(
                f"forbidden top-level path: {top}"
            )

    for raw in (
        FORBIDDEN_EXACT_PATHS
    ):
        if (
            ROOT
            / raw
        ).exists():
            fail(
                f"forbidden path: {raw}"
            )


def check_files() -> None:
    checked = 0

    for path in iter_files():
        rel = relative(
            path
        )

        name_lower = (
            path.name.lower()
        )

        if (
            name_lower
            in FORBIDDEN_SENSITIVE_NAMES
        ):
            fail(
                f"sensitive filename: {rel}"
            )

        if (
            path.suffix.lower()
            in FORBIDDEN_DATA_SUFFIXES
        ):
            fail(
                f"data/result file forbidden: {rel}"
            )

        lowered = (
            rel.lower()
        )

        result_artifact_suffixes = {
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".parquet",
            ".feather",
            ".xlsx",
            ".xls",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
        }

        if (
            path.suffix.lower()
            in result_artifact_suffixes
        ):
            for fragment in (
                FORBIDDEN_NAME_FRAGMENTS
            ):
                if (
                    fragment
                    in lowered
                ):
                    fail(
                        "result-bearing artifact "
                        f"forbidden: {rel}"
                    )

        size = (
            path.stat().st_size
        )

        if (
            size
            > 2_000_000
        ):
            fail(
                f"file exceeds 2 MB: {rel}"
            )

        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            fail(
                f"unexpected binary file: {rel}"
            )

        for pattern in (
            SECRET_PATTERNS
        ):
            if pattern.search(
                text
            ):
                fail(
                    f"possible secret in {rel}"
                )

        checked += 1

    if checked < 20:
        fail(
            "unexpectedly small public repository"
        )

    print(
        f"[OK] audited {checked} files"
    )


def check_license() -> None:
    text = (
        ROOT
        / "LICENSE"
    ).read_text(
        encoding="utf-8"
    )

    if (
        "Apache License"
        not in text
        or "Version 2.0"
        not in text
    ):
        fail(
            "LICENSE is not Apache-2.0"
        )

    print(
        "[OK] Apache-2.0 license"
    )


def main() -> None:
    check_required_files()
    check_forbidden_paths()
    check_files()
    check_license()

    print(
        "[OK] public repository boundary PASS"
    )


if __name__ == "__main__":
    main()
