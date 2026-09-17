# Change control

## Research layer

Frozen empirical results must not be modified retrospectively.

Any additional statistical specification, robustness test or alternative model
introduced after the frozen analysis must be identified as subsequent research
and must not be presented as part of the original preregistered specification.

## Integrity layer

Changes affecting:

- research configuration;
- frozen datasets;
- manifests;
- quantitative outputs;
- inference;
- evidence-pack figures;

must preserve provenance and automated verification.

## Reporting layer

The final evidence pack is generated exclusively from frozen research outputs.

Reporting code must not estimate models, select results or introduce new
statistical tests.

## Repository controls

Before merging a change:

    python -m pytest -q
    git diff --check

The working tree should be clean after the corresponding commit.
