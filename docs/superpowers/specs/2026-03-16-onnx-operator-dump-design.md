# ONNX Operator Dump Design

**Date:** 2026-03-16
**Status:** Historical, superseded by `unified_graph/v1`

This original design predated the shared `op-io-contracts` baseline and
described a legacy `manifest.json` shape with `schema_version`, `nodes`,
`graph_inputs`, `graph_outputs`, nested tensor metadata, and `data_path` fields.

Current output is different:

- `manifest.json` is a `unified_graph/v1` reference graph.
- Top-level keys are only `meta`, `steps`, and `tensors`.
- Tensor files are resolved from the separate flat `tensors/` directory.
- `onnx-dump` is the `ref_graph_producer` role.
- Validation uses `op-io-contracts` package `v0.2.0`, baseline `V1`, and change
  record `changes/v1.0.0-initial-contracts.yaml`.

Current docs:

- `README.md`
- `docs/op-io-contracts-integration.md`
- `examples/basic_add_relu/README.md`
- `../op-io-contracts/baselines/V1.md`
