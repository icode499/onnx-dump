# Basic Add Relu Example

This example builds a small `Add -> Relu` ONNX model, writes deterministic input
tensors, and dumps a checked-in reference output.

```bash
python examples/basic_add_relu/generate.py
```

The generated output is a `unified_graph/v1` reference artifact:

```text
examples/basic_add_relu/output/
├── manifest.json
└── tensors/
    ├── X.npy
    ├── Y.npy
    ├── add_out.npy
    └── Z.npy
```

Validate it with `op-io-contracts` `v0.2.0`:

```bash
op-io-validate graph \
  --graph examples/basic_add_relu/output/manifest.json \
  --tensors examples/basic_add_relu/output/tensors \
  --role ref
```

The relevant contract role is `ref_graph_producer`; from the `onnx-dump`
repository root, the sibling checkout baseline is
`../op-io-contracts/baselines/V1.md`.
