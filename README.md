# Neural competition through functional options

This repository studies one mechanism: training pressure changes the functional
corrections a network can make through shared trainable features, and width can
change whether useful corrections survive or recover within a finite budget.

The maintained model has two primary objects:

- **spectral capability**, derived from the prediction Jacobian, loss curvature,
  optimizer geometry, and an explicit observation budget;
- **residual demand**, which determines which available corrections currently
  matter.

Options are functional response modes, not neurons or teacher-assigned roles.
Their dynamics are modeled with the existing feature–coefficient parameter
evolution and physical ReLU gates. No latent resource state or fitted transition
law is part of the maintained model.

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
