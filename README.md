# Bitcoin Valuation Research

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Public research quality](https://github.com/Hilmar-Corp/btc-valuation-research/actions/workflows/quality.yml/badge.svg)](https://github.com/Hilmar-Corp/btc-valuation-research/actions/workflows/quality.yml)
[![CodeQL](https://github.com/Hilmar-Corp/btc-valuation-research/actions/workflows/codeql.yml/badge.svg)](https://github.com/Hilmar-Corp/btc-valuation-research/actions/workflows/codeql.yml)
[![Dependency and SBOM controls](https://github.com/Hilmar-Corp/btc-valuation-research/actions/workflows/security.yml/badge.svg)](https://github.com/Hilmar-Corp/btc-valuation-research/actions/workflows/security.yml)
[![Release](https://img.shields.io/github/v/release/Hilmar-Corp/btc-valuation-research?display_name=tag&sort=semver)](https://github.com/Hilmar-Corp/btc-valuation-research/releases/latest)

A public quantitative research framework for studying economic valuation anchors for Bitcoin.

This repository contains the methodological and software layer of HilmarCorp research on Bitcoin valuation.

Empirical datasets, research results, publication figures and proprietary HilmarCorp model assets are intentionally not distributed through this repository.

## Research question

Bitcoin does not generate dividends, coupons or contractual cash flows.

The central research question is:

> Do commonly proposed economic anchors for Bitcoin produce valuation relationships that are stable, robust and useful outside the estimation sample?

The framework does not assume that any model-implied value represents a uniquely identified intrinsic value.

## Valuation families

The research framework studies four broad economic approaches:

- scarcity and stock-to-flow;
- network activity;
- electricity-based production cost;
- conditional gold market-share scenarios.

Each framework represents a distinct economic hypothesis.

## Research design

The methodological framework includes:

- monthly data construction;
- explicit data contracts;
- stationarity diagnostics;
- model-admissibility rules;
- HAC inference;
- multiple-testing correction;
- structural-stability analysis;
- expanding pseudo-out-of-sample evaluation;
- naive benchmark comparison;
- dependence-aware bootstrap procedures;
- predictive regressions;
- robustness and falsification analysis.

High historical fit is not treated as sufficient evidence of a valid valuation model.

Predictive association is not interpreted as causality.

Model-implied values are not interpreted as directly observed fundamental values.

## Pseudo-out-of-sample discipline

For each evaluation period, model parameters are estimated using observations strictly prior to the target period.

The target-period Bitcoin price is not used to estimate its own model-implied valuation.

Naive benchmarks remain part of the evaluation framework so that economic models are not assessed in isolation.

## Data sources

The research framework supports data obtained from external providers including:

- Coinbase Exchange;
- Coin Metrics;
- Cambridge Centre for Alternative Finance;
- World Gold Council.

No third-party dataset or source material is distributed through this public repository.

Users wishing to reproduce empirical analyses must obtain the relevant data directly from the respective providers and comply with their applicable terms.

## Public repository scope

This repository may contain:

- research specifications;
- data contracts;
- acquisition code;
- transformation code;
- statistical diagnostics;
- evaluation code;
- integrity controls;
- methodological documentation.

It intentionally excludes:

- raw or processed provider datasets;
- empirical result tables;
- publication figures;
- research-result manifests;
- post-result numerical regression locks;
- Nostra AI model logic;
- production signals;
- proprietary features or parameters;
- client information;
- credentials;
- infrastructure secrets.

## Repository structure

    .
    ├── .github/
    ├── configs/
    ├── src/
    ├── tests/
    ├── tools/
    ├── CHANGE_CONTROL.md
    ├── CITATION.cff
    ├── LICENSE
    ├── METHODOLOGY.md
    ├── NOTICE
    ├── PUBLICATION_SCOPE.md
    ├── REPRODUCIBILITY.md
    ├── SECURITY.md
    ├── THIRD_PARTY_NOTICES.md
    ├── research_spec.md
    └── README.md

## Verification

Run:

    python -m pytest -q

and:

    python tools/audit_public_repository.py

## Proprietary boundary

This repository does not contain Nostra AI production logic, private model parameters, production signals or confidential HilmarCorp research assets.

The public research framework should not be interpreted as a reproduction of HilmarCorp's proprietary allocation technology.

## Licensing

Original HilmarCorp source code, tests, configuration files and technical documentation are licensed under the Apache License, Version 2.0 unless expressly stated otherwise.

Third-party datasets and source materials are not distributed or relicensed under Apache-2.0.

See `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md`.

## Disclaimer

This repository contains quantitative research infrastructure for informational and analytical purposes.

Nothing in this repository constitutes investment advice, portfolio management, execution services, an offer, a solicitation or a recommendation to buy or sell any financial instrument or crypto-asset.

© 2026 HilmarCorp
