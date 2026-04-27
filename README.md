# onnx-dump

Dump per-operator intermediate tensors from ONNX models. The tool produces a
`manifest.json` plus `.npy` files for every captured tensor, which makes it
useful as a golden reference when comparing chip-specific operator behavior.

## Install

```bash
uv sync --extra dev
```

For contract validation, use `op-io-contracts` package `v0.2.0`. This package
release exposes the current V1 contract baseline; the package version
`v0.2.0`, the baseline name `V1`, and the initial change record
`changes/v1.0.0-initial-contracts.yaml` are intentionally separate identifiers.

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

`manifest.json` is a `unified_graph/v1` reference graph document with exactly
these top-level keys:

- `meta`: graph-level format and ONNX opset information
- `steps`: ordered operator steps with names, op types, inputs, outputs, and attributes
- `tensors`: top-level tensor metadata keyed by tensor name

Together with `tensors/*.npy`, the output can be used directly as the `ref`
side input for `op-graph-align compare`.

`onnx-dump`'s shared-contract role is `ref_graph_producer` for
`unified_graph/v1`. The authoritative contract is the `op-io-contracts` V1
baseline:

- `../op-io-contracts/baselines/V1.md`
- `../op-io-contracts/changes/v1.0.0-initial-contracts.yaml`
- `../op-io-contracts/USAGE.md`
- `../op-io-contracts/MIGRATIONS.md`

## Validate the contract

Validate generated output as a reference graph:

```bash
op-io-validate graph \
  --graph output/manifest.json \
  --tensors output/tensors \
  --role ref
```

When testing against a sibling checkout of `op-io-contracts`:

```bash
PYTHONPATH=../op-io-contracts/src uv run --no-sync python -m op_io_contracts.cli graph \
  --graph examples/basic_add_relu/output/manifest.json \
  --tensors examples/basic_add_relu/output/tensors \
  --role ref
```

The Python API returns `ValidationResult`. New code should read
`ValidationResult.issues` for structured `code`, `path`, `contract`,
`baseline`, `roles`, and `docs` fields. Legacy `ValidationResult.errors`
remains available for string-based callers.

On validation failure, the `op-io-validate` CLI prints issue details with
code/path, contract, baseline, affected roles, and docs links.

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
