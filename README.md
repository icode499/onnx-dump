# onnx-dump

Dump per-operator intermediate tensors from ONNX models. The tool produces a
`manifest.json` plus `.npy` files for every captured tensor, which makes it
useful as a golden reference when comparing chip-specific operator behavior.

## Install

```bash
uv sync --extra dev
```

## Python API

```python
from onnx_dump import dump_model

dump_model(
    model_path="model.onnx",
    input_paths=["input_x.npy"],
    output_dir="./output/",
)
```

## CLI

```bash
python -m onnx_dump model.onnx input_x.npy -o ./output/
python -m onnx_dump model.onnx input_x.npy --input-names X -o ./output/
python -m onnx_dump --version
```

## Output

```text
output/
├── manifest.json
└── tensors/
    ├── X.npy
    ├── matmul_out.npy
    └── output.npy
```

`manifest.json` contains operator metadata with the shape, dtype, and data path
for a unified graph document with:

- `meta`: graph-level format and ONNX opset information
- `steps`: ordered operator steps with names, op types, inputs, outputs, and attributes
- `tensors`: top-level tensor metadata keyed by tensor name

Together with `tensors/*.npy`, the output can be used directly as the `ref`
side input for `op-graph-align compare`.

## Run tests

```bash
uv run pytest -v
```

## Example

```bash
python examples/basic_add_relu/generate.py
```

This creates a self-contained example under `examples/basic_add_relu/`:

```text
examples/basic_add_relu/
├── generate.py
├── input/
│   ├── model.onnx
│   └── tensors/
│       ├── X.npy
│       └── Y.npy
└── output/
    ├── manifest.json
    └── tensors/
        ├── X.npy
        ├── Y.npy
        ├── add_out.npy
        └── Z.npy
```
