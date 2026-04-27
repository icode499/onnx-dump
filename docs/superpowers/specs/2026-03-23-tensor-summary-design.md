# Tensor Summary & Preview in manifest.json

**Date:** 2026-03-23
**Status:** Superseded

This draft described adding tensor summaries to a legacy manifest schema with
`schema_version`, nested `inputs`/`outputs`, `graph_inputs`, `graph_outputs`,
and per-tensor `data_path` fields.

That schema is no longer current. `onnx-dump` now emits a
`unified_graph/v1` reference graph with top-level `meta`, `steps`, and
`tensors`, validated by `op-io-contracts` package `v0.2.0` against baseline
`V1`.

Do not implement this draft as written. Any future tensor-summary feature must
first be coordinated through `op-io-contracts` and the active baseline:

- `docs/op-io-contracts-integration.md`
- `../op-io-contracts/baselines/V1.md`
- `../op-io-contracts/changes/v1.0.0-initial-contracts.yaml`
