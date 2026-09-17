# Methodology

This repository accompanies HilmarCorp research on Bitcoin valuation anchors.

The central research question is whether commonly proposed economic anchors for
Bitcoin produce valuation relationships that are stable, robust and useful
outside the estimation sample.

## Valuation families

The research evaluates:

1. scarcity / stock-to-flow;
2. network activity;
3. electricity-based production cost;
4. conditional gold market-share scenarios.

No model-implied value is interpreted as an observable or uniquely identified
fundamental value.

## Frequency and sample

The primary panel uses complete calendar months.

Primary sample:

- September 2017 through August 2026;
- 108 complete monthly observations;
- model-specific samples where required by data availability.

## Econometric discipline

The research includes:

- stationarity diagnostics;
- model admissibility rules;
- HAC inference;
- multiplicity adjustment;
- structural-stability diagnostics;
- expanding pseudo-out-of-sample evaluation;
- naive forecast benchmarks;
- dependence-robust loss comparisons;
- forward-return predictive regressions;
- explicitly labelled post-result falsification tests.

The frozen research specification is recorded in `configs/research.yaml`.

## Interpretation

Historical fit is not treated as evidence of fundamental value.

Predictive association is not treated as evidence of causality.

Gold comparisons are conditional scenario analyses, not intrinsic-value
estimates.
