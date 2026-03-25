# Basic Add Relu Example Layout

**Date:** 2026-03-26
**Status:** Approved

## Problem

The current quick-start example writes all generated files to `/tmp`, which means
the repository does not contain a stable, self-contained example dataset. That
makes the example harder to inspect, harder to share, and awkward to extend when
multiple example types are added later.

## Solution

Replace the single `examples/run_example.py` script with a named example
directory: `examples/basic_add_relu/`.

The example becomes self-contained:

```text
examples/
  basic_add_relu/
    generate.py
    input/
      model.onnx
      tensors/
        X.npy
        Y.npy
    output/
      manifest.json
      tensors/
        X.npy
        Y.npy
        add_out.npy
        Z.npy
```

## Design

### 1. Named Example Directory

Use `basic_add_relu` as the example name. It is specific enough to coexist with
future examples while still being readable to new users.

### 2. Input and Output Separation

The example directory is split into `input/` and `output/`:

- `input/model.onnx` stores the example model
- `input/tensors/` stores model input tensors, named after real graph inputs
- `output/manifest.json` stores exported metadata
- `output/tensors/` stores dumped tensors produced by `dump_model()`

Using `tensors/` on both sides keeps the structure regular across examples.

### 3. Generation Behavior

`generate.py` builds the same `Add -> Relu` model as today, but writes all
artifacts under its own example directory instead of `/tmp`.

Each run recreates the example data in place:

- recreate `input/`
- recreate `output/`
- write deterministic input arrays to `input/tensors/X.npy` and `Y.npy`
- write the model to `input/model.onnx`
- call `dump_model()` with those inputs and `output/`

### 4. Testing

Add a dedicated example test that imports the example generator module and runs
it against a temporary target directory. The test must verify:

- `input/model.onnx` exists
- `input/tensors/X.npy` and `Y.npy` exist
- `output/manifest.json` exists
- `output/tensors/X.npy`, `Y.npy`, `add_out.npy`, and `Z.npy` exist
- dumped tensors match the expected `Add` and `Relu` results

### 5. Documentation

Update the README example section to point to:

```bash
python examples/basic_add_relu/generate.py
```

and describe the example directory layout at a high level.

## Non-Goals

- No new CLI surface for examples
- No changes to the ONNX dumping core logic
- No support for multiple input formats beyond `.npy`
