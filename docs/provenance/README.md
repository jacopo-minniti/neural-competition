# Curation and integrity

- [cleanup.json](cleanup.json): pre-curation Git revision, categorized removals,
  byte counts and table relocations. This is a cleanup record, not study evidence.
- [local_retention.json](local_retention.json): every intentionally retained
  local run/data file at curation, with size and SHA-256. Raw files remain ignored.

Study-specific execution provenance stays beside its [tables](../tables/README.md).
Old audit statements describe their execution time. For the current tree, run
`scripts/check_repository.py`, adding `--local-data` on the original workspace.
See [retention](../data.md) for what cannot be recovered without recomputation.
