# ONNX Operator Dump Tool — Design Spec

**Date:** 2026-03-16
**Status:** Draft

## Purpose

A Python tool that runs an ONNX model with given inputs, captures the intermediate input/output tensors at every operator node, and dumps everything to a structured output directory. The output serves as a "golden reference" so that chip-specific operator implementations can be compared against it for numerical correctness.

## Requirements

- **Primary interface:** Python API (`from onnx_dump import dump_model`), callable from other tools programmatically
- **Secondary interface:** CLI wrapper (`python -m onnx_dump`) for convenience
- **Input format:** `.onnx` model file + `.npy` files for input tensors
- **Output format:** `manifest.json` (metadata) + per-tensor `.npy` files
- **Execution backend:** ONNX Runtime (CPU execution provider)
- **Scope:** Top-level graph operators only (no subgraph recursion for If/Loop/Scan)
- **Function:** Dump only (no built-in comparison mode)
- **Environment management:** uv

## Architecture

### Approach: Graph Surgery

Modify the ONNX graph in-memory to expose every intermediate tensor as a graph output, run inference once, and collect all intermediate values from the result dict.

**Why this approach:**
- Single inference call — simple and efficient
- No dependency on undocumented ORT APIs
- Well-established pattern in the ONNX ecosystem
- The `onnx` library makes it straightforward to add intermediate tensors as graph outputs

### Core Algorithm

**Step 1 — Validate and load the model:**
1. Load the ONNX model via `onnx.load()`
2. Validate the model with `onnx.checker.check_model()` to catch malformed models early

**Step 2 — Augment the graph (graph surgery):**
1. Walk all nodes in `graph.node`, collect every intermediate tensor name (outputs of each node)
2. For each intermediate tensor, add a `ValueInfoProto` to `graph.output` so ONNX Runtime will return it
3. Attempt `onnx.shape_inference.infer_shapes()` to populate type/shape info. If shape inference fails (custom ops, dynamic shapes, opset gaps), fall back to adding outputs with empty type info — ONNX Runtime can handle this for most operators. Shape and dtype will be populated from actual numpy results post-inference.

**Step 3 — Build initializer lookup table:**
1. Iterate `graph.initializer`, convert each `TensorProto` to numpy via `onnx.numpy_helper.to_array()`
2. Store in a dict keyed by tensor name — this is the lookup table for weight/bias/constant tensors

**Step 4 — Load inputs and run inference:**
1. Load each `.npy` input file with `numpy.load()`
2. Map inputs to model input names. Rules:
   - If `input_names` is provided, it must be the same length as `input_paths` (raise error otherwise)
   - If `input_names` is None, map by declaration order, skipping model inputs that have a corresponding initializer (since those don't need user-provided data)
   - It is valid to provide fewer inputs than the model declares, as long as all non-initializer inputs are covered
3. Create an `onnxruntime.InferenceSession` with the augmented model (CPU execution provider)
4. Run inference — the result dict now contains all intermediate tensors plus the original outputs

**Step 5 — Map tensors back to operators:**
For each node in `graph.node`:
- Record: `op_type`, `name`, `input` names, `output` names, `attributes`
- Look up each input/output tensor name in: (1) the inference result dict, (2) user-provided input arrays, (3) the initializer lookup table from Step 3
- All shapes and dtypes in the manifest come from the **actual runtime numpy arrays**, not from static graph type info. This correctly handles dynamic/symbolic dimensions.

**Step 6 — Export:**
- Write each tensor as a `.npy` file under `<output_dir>/tensors/`
- Write `manifest.json` with per-operator records
- If the output directory already exists, overwrite its contents (no `--force` flag needed; the tool always writes fresh output)

## Project Structure

```
data-compare/
├── pyproject.toml           # uv-compatible, dependencies, entry point
├── README.md                # Usage docs with examples
├── examples/
│   ├── simple_add.onnx      # Tiny example model
│   ├── input_x.npy          # Example input
│   └── run_example.sh       # Quick-start script
├── src/
│   └── onnx_dump/
│       ├── __init__.py      # Exports dump_model()
│       ├── cli.py           # Thin CLI wrapper (argparse)
│       ├── graph.py         # Graph surgery: load model, add intermediate outputs
│       ├── runner.py        # Inference execution, collect all tensors
│       └── exporter.py      # Write JSON manifest + .npy files
└── tests/
    ├── test_graph.py
    ├── test_runner.py
    ├── test_exporter.py
    └── test_system.py       # End-to-end accuracy test
```

**Dependencies:** `onnx`, `onnxruntime`, `numpy`

## Python API

```python
from onnx_dump import dump_model

dump_model(
    model_path="model.onnx",
    input_paths=["input_x.npy", "input_y.npy"],
    output_dir="./onnx_dump_output/",
    input_names=None,  # Optional: explicit name mapping; if None, uses declaration order
)
```

## CLI Interface

```bash
python -m onnx_dump model.onnx input_x.npy input_y.npy -o ./output/
python -m onnx_dump model.onnx input_x.npy --input-names X -o ./output/
```

- Positional arg 1: path to `.onnx` model
- Positional args 2+: input `.npy` files
- `-o / --output-dir`: output directory (default: `./onnx_dump_output/`)
- `--input-names`: optional comma-separated list to map `.npy` files to specific model input names
- `--version`: print tool version and exit

## Output Format

### Directory structure
```
<output_dir>/
├── manifest.json
└── tensors/
    ├── X.npy
    ├── Add_0_output.npy
    ├── Relu_0_output.npy
    └── ...
```

### manifest.json schema
```json
{
  "schema_version": "1.0",
  "model_path": "model.onnx",
  "opset_version": 13,
  "nodes": [
    {
      "name": "Add_0",
      "op_type": "Add",
      "domain": "",
      "attributes": {},
      "inputs": [
        {
          "name": "X",
          "shape": [1, 3, 224, 224],
          "dtype": "float32",
          "data_path": "tensors/X.npy"
        }
      ],
      "outputs": [
        {
          "name": "Add_0_output",
          "shape": [1, 3, 224, 224],
          "dtype": "float32",
          "data_path": "tensors/Add_0_output.npy"
        }
      ]
    }
  ],
  "graph_inputs": [
    {"name": "X", "shape": [1, 3, 224, 224], "dtype": "float32", "data_path": "tensors/X.npy"}
  ],
  "graph_outputs": [
    {"name": "Y", "shape": [1, 1000], "dtype": "float32", "data_path": "tensors/Y.npy"}
  ]
}
```

**Field descriptions:**
- `schema_version`: Manifest format version for downstream consumers to detect breaking changes
- `opset_version`: The opset version for the default domain (`""`). If the model imports multiple opsets, only the default domain version is recorded here.
- `nodes[].name`: Operator name from the graph. If the node has no name, auto-generated as `<index>_<op_type>` where `index` is the node's positional index in `graph.node` (e.g., `0_Conv`, `1_Relu`, `2_Conv`). This scheme is used consistently everywhere.
- `nodes[].op_type`: ONNX operator type (e.g., Conv, Relu, MatMul)
- `nodes[].domain`: The ONNX domain this operator belongs to (e.g., `""` for default, `"ai.onnx.ml"` for ML operators)
- `nodes[].attributes`: Operator-specific parameters serialized to JSON. Supported attribute types: int, float, string, and lists thereof are serialized directly. Tensor attributes are serialized as `{"shape": [...], "dtype": "...", "data_path": "tensors/..."}` (saved as .npy). Graph attributes are omitted (value: `null`).
- `nodes[].inputs/outputs`: Each tensor includes `name`, `shape`, `dtype`, and `data_path` (path relative to the output directory)
- `graph_inputs/graph_outputs`: Model-level I/O for context

## Error Handling

1. **Unnamed nodes:** Auto-generate names using `<index>_<op_type>` where `index` is the positional index in `graph.node` (e.g., `0_Conv`, `1_Relu`)
2. **Constant/Initializer inputs:** Extract from `graph.initializer` via `onnx.numpy_helper.to_array()`. Saved as `.npy` files and referenced in the manifest like any other tensor.
3. **Input count mismatch:** Raise clear error with expected vs. provided input names. If `input_names` is provided, it must match `input_paths` in length. Model inputs backed by initializers do not require user-provided data.
4. **Duplicate tensor names:** By default, use the bare tensor name as the `.npy` filename (e.g., `Add_0_output.npy`). If two tensors share the same name (rare but possible), fall back to `<node_name>__<tensor_name>.npy` (double underscore separator) for the duplicates. All `.npy` files live directly in `tensors/` with no subdirectories.
5. **Shared tensors:** If a single tensor is consumed as input by multiple nodes, the `.npy` file is written once. All manifest entries referencing that tensor point to the same `data_path`. No duplicate writes.
6. **Large models:** Log warning after inference if total dumped tensor size exceeds 10GB (based on actual numpy array sizes).
7. **Existing output directory:** Overwrite contents. No `--force` flag needed.
8. **Constant nodes:** `Constant` operator nodes are treated like any other operator — their output tensors are captured through inference and dumped normally.

## Logging

Use Python's `logging` module with a named logger: `logging.getLogger("onnx_dump")`. This allows programmatic callers to configure log levels without interfering with their own logging. The CLI wrapper sets up a basic handler; the Python API leaves logging configuration to the caller.

## Testing Strategy

### Unit tests
- **`test_graph.py`:** Create a tiny ONNX model programmatically (e.g., `Add` two inputs), verify graph surgery adds all intermediate tensors as outputs. Also test auto-naming for unnamed nodes.
- **`test_runner.py`:** Run the augmented model, verify all expected tensor names appear in results with correct shapes. Include a test with initializer-backed inputs.
- **`test_exporter.py`:** Verify manifest.json schema is correct, `.npy` files are loadable and match expected values. Include a test for duplicate tensor name disambiguation (flat `__` naming).

### Integration test
- Build a small multi-operator model (e.g., `Conv -> Relu -> MaxPool`), run the full pipeline, verify the output directory contains correct manifest and tensor files

### System test (`test_system.py`) — End-to-end accuracy
- Build a multi-operator model (e.g., `MatMul -> Add -> Relu`)
- Compute the expected output of each operator manually using numpy (`np.matmul()`, `np.add()`, `np.maximum(0, x)`)
- Run the full `dump_model()` pipeline
- Load each dumped `.npy` tensor and compare against manually computed expected values using `np.allclose()` with appropriate tolerances
- This verifies the tool produces **numerically correct** intermediate values, ensuring trustworthiness as a golden reference

### Test fixtures
- Generated programmatically using `onnx.helper` — no shipped binary model files
