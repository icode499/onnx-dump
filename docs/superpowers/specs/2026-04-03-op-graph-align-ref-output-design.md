# ONNX Dump Ref-Side Output Alignment

**Date:** 2026-04-03
**Status:** Implemented and superseded by the shared V1 contract baseline

This design moved `onnx-dump` from the legacy debug manifest shape
(`schema_version`, `nodes`, `graph_inputs`, `graph_outputs`, nested tensor
records, and `data_path`) to the shared `unified_graph/v1` reference graph
shape.

Current user-facing guidance lives in:

- `README.md`
- `docs/op-io-contracts-integration.md`
- `examples/basic_add_relu/README.md`
- `../op-io-contracts/baselines/V1.md`
- `../op-io-contracts/changes/v1.0.0-initial-contracts.yaml`

## Current Decisions

- `manifest.json` remains the graph file name.
- The JSON document is `unified_graph/v1` with only `meta`, `steps`, and
  `tensors` at the top level.
- `onnx-dump` is the `unified_graph/v1` `ref_graph_producer`.
- Validation uses `op-io-contracts` package `v0.2.0` against baseline `V1`.
- The initial contract change record is
  `changes/v1.0.0-initial-contracts.yaml`.
- Tensor files remain flat under `tensors/<tensor_name>.npy`.
- Tensor metadata lives in top-level `tensors`; tensor entries do not carry
  `data_path`.

## Validation

```bash
op-io-validate graph \
  --graph output/manifest.json \
  --tensors output/tensors \
  --role ref
```

For local source validation:

```bash
PYTHONPATH=../op-io-contracts/src uv run --no-sync python -m op_io_contracts.cli graph \
  --graph examples/basic_add_relu/output/manifest.json \
  --tensors examples/basic_add_relu/output/tensors \
  --role ref
```

`ValidationResult.issues` is the primary structured error API. Legacy
`ValidationResult.errors` remains available for string-based callers.
