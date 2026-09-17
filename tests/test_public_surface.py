from __future__ import annotations

import compileall
from pathlib import Path

import yaml


ROOT = Path(
    __file__
).resolve().parents[1]


def test_forbidden_result_directories_absent():
    for name in [
        "data",
        "outputs",
        "manifests",
    ]:
        assert not (
            ROOT / name
        ).exists()


def test_post_results_lock_absent():
    assert not (
        ROOT
        / "configs"
        / "post_results_regression_lock.yaml"
    ).exists()


def test_reporting_layer_absent():
    assert not (
        ROOT
        / "src"
        / "reporting"
    ).exists()


def test_required_public_documents_exist():
    required = [
        "README.md",
        "LICENSE",
        "NOTICE",
        "METHODOLOGY.md",
        "REPRODUCIBILITY.md",
        "PUBLICATION_SCOPE.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "CHANGE_CONTROL.md",
        "CITATION.cff",
        "research_spec.md",
    ]

    for name in required:
        assert (
            ROOT / name
        ).exists(), name


def test_research_yaml_parses():
    path = (
        ROOT
        / "configs"
        / "research.yaml"
    )

    assert path.exists()

    data = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(
        data,
        dict,
    )

    assert data


def test_data_contract_yaml_parses():
    path = (
        ROOT
        / "configs"
        / "data_contract.yaml"
    )

    assert path.exists()

    data = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(
        data,
        dict,
    )

    assert data


def test_public_python_sources_compile():
    assert compileall.compile_dir(
        ROOT / "src",
        quiet=1,
        force=True,
    )
