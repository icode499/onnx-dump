# ONNX Dump Ref-Side Output Alignment

**Date:** 2026-04-03
**Status:** Draft

## Problem

`onnx-dump` currently writes a debug-oriented `manifest.json` with nested
`nodes`, `graph_inputs`, and `graph_outputs` records. That format is useful for
inspection, but it is not the same contract required by
`docs/op_graph_align_input_spec.md` for the `ref` side of
`op-graph-align compare`.

The target workflow is stricter:

- Keep the output directory shape as `manifest.json + tensors/*.npy`
- Redefine `manifest.json` so its contents match the `ref` unified graph schema
- Make the output usable as `--ref-graph` plus `--ref-tensors` without manual
  conversion

## Approved Decisions

- `manifest.json` remains the top-level graph file name
- The JSON content becomes the `ref` unified graph schema, not the legacy
  manifest schema
- The default goal is direct compatibility with `op-graph-align compare` on the
  `ref` side
- Tensor `dtype` strings use ONNX-style names such as `FLOAT` and `INT64`
- `storage_format` defaults to `"plain"` when the tool can safely assert it
- `layout` and `quant` are omitted unless the tool can determine them reliably

## Goals

- Produce a `manifest.json` that satisfies the `ref`-side structural contract:
  top-level `meta`, `steps`, and `tensors`
- Preserve the existing `dump_model(...)` API and CLI invocation shape
- Preserve flat tensor file output under `tensors/<tensor_name>.npy`
- Cover graph inputs, outputs, intermediate tensors, and initializers so the
  exported tensors directory is complete enough for comparison
- Fail fast on export states that would produce a misleading or partially
  unusable `ref` artifact

## Non-Goals

- No support for the `chip`-side graph format or config generation in this work
- No attempt to infer `layout` or `quant` from weak heuristics
- No backwards-compatible legacy manifest mode in the default output path
- No expansion into subgraph recursion for `If`, `Loop`, or `Scan`

## Proposed Design

### 1. Internal Structure

Refactor the export path into two explicit layers:

1. Collection layer
   - Keep the current responsibilities for loading the model, augmenting outputs,
     running ONNX Runtime, and collecting tensors from inputs, inference
     results, and initializers
2. Ref graph builder layer
   - Convert collected ONNX/runtime facts into an in-memory unified graph
     document with `meta`, `steps`, and `tensors`
   - Normalize tensor metadata into the `op-graph-align` `ref` contract
3. Export layer
   - Write the unified graph document to `manifest.json`
   - Write every tensor array to `tensors/<tensor_name>.npy`

This split keeps schema decisions out of the file-writing path and makes the new
contract testable without depending on disk I/O.

### 2. Output Schema

`manifest.json` will use this structure:

```json
{
  "meta": {
    "format_version": 1,
    "graph_spec": "onnx",
    "opset_version": 17
  },
  "steps": [
    {
      "id": "Add_0",
      "name": "Add_0",
      "op_type": "Add",
      "inputs": ["X", "Y"],
      "outputs": ["add_out"],
      "attributes": {}
    }
  ],
  "tensors": {
    "X": {
      "dtype": "FLOAT",
      "shape": [2, 3],
      "storage_format": "plain"
    }
  }
}
```

Field rules:

- `meta.format_version` is fixed at `1`
- `meta.graph_spec` is fixed at `"onnx"`
- `meta.opset_version` is the imported opset for the default ONNX domain
- `steps` preserves the ONNX node order
- `steps[].id` and `steps[].name` both use the stable node name
- `tensors` is keyed by tensor name and covers every tensor referenced by the
  exported graph

### 3. Step Construction

Each ONNX node becomes one `steps[]` entry:

- `id`: node name after the existing auto-naming pass for unnamed nodes
- `name`: same as `id`
- `op_type`: `node.op_type`
- `inputs`: `node.input` names in original order, excluding empty names
- `outputs`: `node.output` names in original order, excluding empty names
- `attributes`: JSON-safe scalar or scalar-array values only

The builder will not carry over the legacy nested tensor metadata in
`inputs`/`outputs`. Tensor metadata now lives only in the top-level `tensors`
object, which is what `op-graph-align` expects.

### 4. Tensor Coverage and Metadata

The builder forms a complete tensor table from:

- model inputs provided by the caller
- initializer tensors extracted from the ONNX graph
- graph outputs and intermediate outputs returned by ONNX Runtime

For every tensor name used by `steps[].inputs` or `steps[].outputs`, the output
must contain:

- `tensors.<name>`
- `tensors/<name>.npy`

Each tensor metadata entry follows these rules:

- `dtype`
  - Prefer ONNX type information when available
  - Fall back to mapping the actual NumPy dtype to an ONNX-style name
  - Raise an error if no stable mapping exists
- `shape`
  - Use the actual dumped array shape
  - This keeps the declared metadata aligned with what downstream consumers will
    load from disk
- `storage_format`
  - Always emit `"plain"` for tensors produced by this tool in this phase
- `layout`
  - Omit by default
- `quant`
  - Omit by default

The output does not need a `data_path` field in `tensors`. The graph file and
tensor directory are already supplied separately to `op-graph-align`, and the
target spec resolves tensor data by `tensor_name + ".npy"`.

### 5. Attribute Normalization

The current exporter serializes some ONNX attribute forms into structures that
do not satisfy the `ref` schema. The new builder normalizes attributes into the
allowed subset:

- allowed as-is:
  - strings
  - numbers
  - booleans
  - `null`
  - arrays of those scalar values
- normalized:
  - `GRAPH` and `GRAPHS` attributes become `null`
- rejected from the output shape:
  - nested objects
  - arrays containing objects
  - tensor-attribute payload objects with `shape`, `dtype`, `data_path`

For tensor-valued ONNX attributes, this phase will not try to invent a richer
schema. The attribute is reduced to `null` rather than emitting a non-compliant
object.

### 6. Error Handling

Export fails instead of silently producing partial `ref` artifacts when:

- a step references a tensor that has neither dumped array data nor sufficient
  metadata to build a valid tensor entry
- a tensor name cannot be written safely as `tensors/<tensor_name>.npy`
- the tool cannot map a tensor dtype to a stable ONNX-style string
- the builder would otherwise emit a manifest that violates the required
  top-level schema

Existing behavior that masked schema problems will be removed:

- no duplicate-file-name fallback such as `<node_name>__<tensor_name>.npy`
- no legacy nested tensor records under `steps`
- no NumPy-style dtype strings in the new manifest

This is an intentional breaking change at the JSON-content level, even though
the file name remains `manifest.json`.

## Implementation Notes

### Module Changes

- `src/onnx_dump/exporter.py`
  - shrink to file-writing responsibilities
  - write unified graph JSON instead of the legacy manifest shape
- `src/onnx_dump/graph.py`
  - keep model augmentation and initializer extraction
  - optionally expose helper utilities that make ONNX type metadata easier to
    consume in the builder
- new builder module, likely `src/onnx_dump/ref_graph.py`
  - build `meta`
  - build `steps`
  - build `tensors`
  - normalize attributes
  - map NumPy/ONNX dtypes into the target strings
- `src/onnx_dump/__init__.py`
  - keep `dump_model(...)` signature unchanged
  - route collected data through the new builder before export

### README and Example Updates

Update repository documentation and example output expectations so they describe
the new schema:

- README output section
- example tree and example `manifest.json`
- any tests or docs that still describe `nodes`, `graph_inputs`, or
  `graph_outputs`

## Testing Strategy

### Builder Unit Tests

Add focused tests for the new unified graph builder:

- builds `meta` with `format_version=1`, `graph_spec="onnx"`, and ONNX opset
- emits `steps` with stable `id`, `name`, ordered `inputs`, and ordered
  `outputs`
- emits `tensors` for graph inputs, intermediates, graph outputs, and
  initializers
- maps tensor dtypes to ONNX-style strings
- uses real dumped array shapes in `tensors`
- sets `storage_format` to `"plain"`
- reduces unsupported ONNX attributes to compliant JSON values

### Exporter Tests

Revise exporter tests so they validate the new contract:

- `manifest.json` contains `meta`, `steps`, and `tensors`
- no legacy `nodes`, `graph_inputs`, or `graph_outputs` sections remain
- every tensor in the manifest has a corresponding `.npy` file in `tensors/`
- dumped `.npy` files still round-trip through `numpy.load()`

### End-to-End Coverage

Update system/example tests to verify the output is directly usable as a
`ref`-side artifact:

- `meta.graph_spec == "onnx"`
- `meta.opset_version` exists
- every tensor referenced in `steps` exists in top-level `tensors`
- every tensor declared in `tensors` has a same-name `.npy` file
- example manifests now show ONNX-style `dtype` values

## Risks and Mitigations

- Risk: downstream users may still consume the old debug manifest schema
  - Mitigation: document the breaking change clearly in README and tests; keep
    the file name stable but update the schema description everywhere
- Risk: some ONNX models may expose dtypes or attributes that do not map cleanly
  - Mitigation: fail clearly and keep the mapping table explicit and test-driven
- Risk: actual runtime arrays may differ from static inferred shapes
  - Mitigation: declare and test that runtime arrays are the source of truth for
    exported tensor `shape`

## Open Questions Resolved

- File name remains `manifest.json`
- Legacy manifest compatibility is not preserved in the default path
- The tool targets direct `ref`-side usability, not merely schema resemblance
- `storage_format` defaults to `"plain"`; `layout` and `quant` remain omitted
  unless a future phase can source them reliably
