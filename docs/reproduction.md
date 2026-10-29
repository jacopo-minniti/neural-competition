# Reproduction

Run from the repository root.

## Validate the maintained repository

```sh
.venv/bin/python scripts/check_repository.py
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
```

## Regenerate compact analyses

The retained evidence directories contain aggregate records but not all raw
inputs. Full regeneration therefore starts with a fresh output directory using
the corresponding producer, followed by its analyzer. Producers refuse to
overwrite existing protocols.

Principal entry points:

```sh
# Heterogeneous toy tasks and causal resource probes
.venv/bin/python scripts/studies/run_toy_resource_competition.py --help

# N-refinement frequency hierarchies
.venv/bin/python scripts/studies/run_refinement_hierarchy.py --help

# Delayed-demand recovery study
.venv/bin/python scripts/studies/probe_delayed_option_recovery.py --help

# MNIST local response and finite-budget causal probes
.venv/bin/python scripts/studies/train_response_checkpoints.py --help
.venv/bin/python scripts/studies/probe_causal_response.py --help
```

Exact historical commands and source snapshots are recorded in the evidence
READMEs and protocols. To inspect the complete pre-curation workspace, use a
separate checkout of commit `92b7904`.
