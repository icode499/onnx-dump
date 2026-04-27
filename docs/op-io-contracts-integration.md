# op-io-contracts Integration

`onnx-dump` produces reference-side `unified_graph/v1` artifacts and therefore
acts as a `ref_graph_producer`.

## Current Contract State

- Package release: `op-io-contracts` `v0.2.0`
- Active baseline: `V1`
- Baseline document: `../op-io-contracts/baselines/V1.md`
- Initial change record: `../op-io-contracts/changes/v1.0.0-initial-contracts.yaml`
- Usage guide: `../op-io-contracts/USAGE.md`
- Migration guide: `../op-io-contracts/MIGRATIONS.md`

The package version and baseline are different concepts. `v0.2.0` is the
Python package release that includes structured validation issues and CLI
rendering. `V1` is the file-contract baseline for `unified_graph/v1` and
`diagnose/v1`.

## Artifact Shape

`onnx-dump` writes:

```text
output/
├── manifest.json
└── tensors/
    └── <tensor_name>.npy
```

`manifest.json` must be a `unified_graph/v1` JSON object with only:

- `meta`
- `steps`
- `tensors`

For ONNX reference graphs, `meta.graph_spec` is `onnx`,
`meta.format_version` is `1`, and `meta.opset_version` is an integer.

## Validation

Use the graph validator with `role="ref"`:

```bash
op-io-validate graph \
  --graph output/manifest.json \
  --tensors output/tensors \
  --role ref
```

For local source testing without changing this repository's dependency files:

```bash
PYTHONPATH=../op-io-contracts/src uv run --no-sync python -m op_io_contracts.cli graph \
  --graph examples/basic_add_relu/output/manifest.json \
  --tensors examples/basic_add_relu/output/tensors \
  --role ref
```

In Python tests:

```python
from op_io_contracts import validate_unified_graph

result = validate_unified_graph(
    "output/manifest.json",
    tensor_dir="output/tensors",
    role="ref",
)

if not result.ok:
    for issue in result.issues:
        print(issue.code, issue.path, issue.contract, issue.baseline, issue.roles, issue.docs)
```

`ValidationResult.issues` is the primary structured API. Each issue exposes
`code`, `path`, `message`, `contract`, `baseline`, `roles`, `change_id`, and
`docs`. `ValidationResult.errors` remains available for legacy string-based
callers.

On failure, the CLI prints structured issue details including code/path,
contract, baseline, affected roles, and docs links.
