# ONNX Operator Dump Implementation Plan

**Date:** 2026-03-17
**Status:** Historical, superseded by `unified_graph/v1`

This long implementation plan targeted the original debug manifest format and
included outdated code snippets, tests, README text, and example commands for
`schema_version`, `nodes`, `graph_inputs`, `graph_outputs`, nested tensor
records, and `data_path`.

The current flow is governed by `op-io-contracts`:

- Package release: `v0.2.0`
- Baseline: `V1`
- Change record: `changes/v1.0.0-initial-contracts.yaml`
- Contract: `unified_graph/v1`
- `onnx-dump` role: `ref_graph_producer`

Use the current docs instead:

- `README.md`
- `docs/op-io-contracts-integration.md`
- `examples/basic_add_relu/README.md`
- `../op-io-contracts/baselines/V1.md`
