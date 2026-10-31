# Neural competition through functional options

This repository studies one mechanism: training pressure changes the functional
corrections a network can make through shared trainable features, and width can
change whether useful corrections survive or recover within a finite budget.

The maintained model has two primary objects:

- **spectral capability**, derived from the prediction Jacobian, loss curvature,
  optimizer geometry, and an explicit observation budget;
- **residual demand**, which determines which available corrections currently
  matter.

An option is a correction required by a declared residual, considered together
with the network's response to it. Spectral directions describe capability modes;
their residual loading specifies the options. An option is closed when demand remains but its local
response is zero; satisfying demand is not closure. The precise spectral
definition and unchanged value formulas are in the
[spectral derivation](docs/model/spectral_options.md#3-capability-modes-and-demanded-options).
Their dynamics are modeled with the existing feature–coefficient parameter
evolution and physical ReLU gates. The current
[reserve model](docs/model/competition_reserve.md) distinguishes present
capability from response protected against common-task learning. It derives an
exact state certificate, a completion curve indexed by common learning rather
than elapsed steps, and a restricted last-alternative loss law. Tests use 24–32
power-law refinements, with continuous context variation and sufficient narrow
capacity. The earlier [conditional forecasting model](docs/model/power_law_competition.md)
remains a numerical tool; the [Gaussian pressure law](docs/model/probabilistic_competition.md)
is a more restricted analytic special case.
No latent resource state or fitted transition law is part of the maintained model.

## Reading order

1. [Research objective](OBJECTIVE.md)
2. [Current research context](docs/research_context.md)
3. [Model overview](docs/model/README.md)
4. [Experiment overview](docs/experiments/README.md)
5. [Reproduction](docs/reproduction.md) and [data retention](docs/data.md)

The complete repository state before this curation is preserved in Git commit
`92b7904` (`Snapshot complete options research before simplification`).

## Quick validation

```sh
.venv/bin/python scripts/check_repository.py
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```
