# ONNX Operator Dump Tool Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python tool that runs an ONNX model, captures every operator's input/output tensors via graph surgery, and dumps them as `.npy` files with a JSON manifest — serving as a golden reference for chip-specific operator comparison.

**Architecture:** Graph surgery approach — modify the ONNX graph in-memory to expose all intermediate tensors as graph outputs, run inference once with ONNX Runtime, collect all values from the result dict. Exports a `manifest.json` plus per-tensor `.npy` files.

**Tech Stack:** Python, uv, onnx, onnxruntime, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-onnx-operator-dump-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, dependencies, CLI entry point, pytest config |
| `src/onnx_dump/__init__.py` | Public API: exports `dump_model()` and `__version__` |
| `src/onnx_dump/graph.py` | Load & validate ONNX model, graph surgery (add intermediate outputs), build initializer table |
| `src/onnx_dump/runner.py` | Map user inputs to model inputs, run ORT inference, collect all tensors |
| `src/onnx_dump/exporter.py` | Write manifest.json + .npy tensor files, handle naming/dedup |
| `src/onnx_dump/cli.py` | argparse CLI wrapper over `dump_model()` |
| `src/onnx_dump/__main__.py` | Enables `python -m onnx_dump` |
| `tests/conftest.py` | Shared test fixtures (ONNX model builders) |
| `tests/test_graph.py` | Tests for graph surgery and initializer extraction |
| `tests/test_runner.py` | Tests for input mapping and inference execution |
| `tests/test_exporter.py` | Tests for manifest schema and .npy file output |
| `tests/test_system.py` | End-to-end accuracy test comparing dumped tensors to numpy reference |
| `tests/test_cli.py` | CLI argument parsing and integration |
| `README.md` | Usage documentation with examples |
| `examples/run_example.py` | Quick-start example script |

---

## Chunk 1: Project Scaffold and Graph Surgery

### Task 1: Project setup with uv

**Files:**
- Create: `pyproject.toml`
- Create: `src/onnx_dump/__init__.py`

- [ ] **Step 1: Initialize project structure**

```bash
cd /home/hyz/code/prodEng/data-compare
uv init --lib --name onnx-dump
# Clean up uv-generated files that will be replaced
rm -f src/onnx_dump/py.typed src/onnx_dump/__init__.py 2>/dev/null || true
```

- [ ] **Step 2: Replace pyproject.toml with correct config**

```toml
[project]
name = "onnx-dump"
version = "0.1.0"
description = "Dump per-operator intermediate tensors from ONNX models"
requires-python = ">=3.10"
dependencies = [
    "onnx>=1.14.0",
    "onnxruntime>=1.16.0",
    "numpy>=1.24.0",
]

[project.scripts]
onnx-dump = "onnx_dump.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/onnx_dump"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[project.optional-dependencies]
dev = ["pytest>=7.0.0"]
```

- [ ] **Step 3: Create src/onnx_dump/__init__.py**

```python
"""ONNX Operator Dump Tool — capture per-operator intermediate tensors."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.egg-info/
.venv/
dist/
build/
*.pyc
```

- [ ] **Step 5: Install dependencies**

```bash
uv sync
```

- [ ] **Step 6: Verify installation**

```bash
uv run python -c "import onnx_dump; print(onnx_dump.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml src/onnx_dump/__init__.py .gitignore
git commit -m "chore: initialize onnx-dump project with uv"
```

---

### Task 2: Graph surgery — load, validate, augment

**Files:**
- Create: `src/onnx_dump/graph.py`
- Create: `tests/conftest.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Create shared test fixtures in conftest.py**

```python
"""Shared test fixtures — programmatic ONNX model builders."""

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest


@pytest.fixture
def simple_add_model():
    """Model: Z = X + Y (two inputs, one Add node, one output)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])

    add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="Add_0")

    graph = helper.make_graph([add_node], "simple_add", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def multi_op_model():
    """Model: X -> MatMul(W) -> Add(B) -> Relu -> output.
    W and B are initializers."""
    rng = np.random.RandomState(123)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    W_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 3],
                                 rng.randn(4, 3).astype(np.float32).flatten().tolist())
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [3],
                                 rng.randn(3).astype(np.float32).flatten().tolist())

    matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="MatMul_0")
    add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"], name="Add_0")
    relu = helper.make_node("Relu", ["add_out"], ["output"], name="Relu_0")

    graph = helper.make_graph(
        [matmul, add, relu], "multi_op",
        [X], [out],
        initializer=[W_init, B_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def unnamed_nodes_model():
    """Model with unnamed nodes: X -> Abs -> Neg -> output."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    abs_node = helper.make_node("Abs", ["X"], ["abs_out"])  # no name
    neg_node = helper.make_node("Neg", ["abs_out"], ["output"])  # no name

    graph = helper.make_graph([abs_node, neg_node], "unnamed", [X], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model
```

- [ ] **Step 2: Write failing tests for graph surgery**

`tests/test_graph.py`:

```python
"""Tests for graph.py — model loading, validation, graph surgery, initializer extraction."""

import onnx
import numpy as np
from onnx_dump.graph import load_and_augment, build_initializer_table


class TestLoadAndAugment:
    def test_adds_intermediate_outputs(self, simple_add_model):
        """Graph surgery should add intermediate tensor Z as an output."""
        augmented, original_outputs = load_and_augment(simple_add_model)
        output_names = [o.name for o in augmented.graph.output]
        # Original output Z should still be there
        assert "Z" in output_names
        assert original_outputs == {"Z"}

    def test_multi_op_intermediates(self, multi_op_model):
        """All intermediate tensors (matmul_out, add_out) should be in outputs."""
        augmented, original_outputs = load_and_augment(multi_op_model)
        output_names = [o.name for o in augmented.graph.output]
        assert "matmul_out" in output_names
        assert "add_out" in output_names
        assert "output" in output_names
        assert original_outputs == {"output"}

    def test_auto_names_unnamed_nodes(self, unnamed_nodes_model):
        """Unnamed nodes should get <index>_<op_type> names."""
        augmented, _ = load_and_augment(unnamed_nodes_model)
        node_names = [n.name for n in augmented.graph.node]
        assert node_names == ["0_Abs", "1_Neg"]

    def test_load_from_path(self, tmp_path, simple_add_model):
        """Should accept a file path string."""
        model_path = str(tmp_path / "model.onnx")
        onnx.save(simple_add_model, model_path)
        augmented, _ = load_and_augment(model_path)
        output_names = [o.name for o in augmented.graph.output]
        assert "Z" in output_names


class TestBuildInitializerTable:
    def test_extracts_initializers(self, multi_op_model):
        """Should extract W and B as numpy arrays."""
        table = build_initializer_table(multi_op_model)
        assert "W" in table
        assert "B" in table
        assert table["W"].shape == (4, 3)
        assert table["B"].shape == (3,)
        assert table["W"].dtype == np.float32

    def test_empty_for_no_initializers(self, simple_add_model):
        """Model without initializers should return empty dict."""
        table = build_initializer_table(simple_add_model)
        assert table == {}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'onnx_dump.graph'`

- [ ] **Step 4: Implement graph.py**

`src/onnx_dump/graph.py`:

```python
"""Graph surgery — load, validate, and augment ONNX models."""

import logging
from typing import Union

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper, shape_inference

logger = logging.getLogger("onnx_dump")


def load_and_augment(model_or_path: Union[str, ModelProto]) -> tuple[ModelProto, set[str]]:
    """Load an ONNX model, validate it, name unnamed nodes, and add all
    intermediate tensors as graph outputs.

    Args:
        model_or_path: Path to .onnx file or an in-memory ModelProto.

    Returns:
        Tuple of (augmented ModelProto, set of original output names).
    """
    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path)
    else:
        model = model_or_path

    onnx.checker.check_model(model)

    graph = model.graph

    # Auto-name unnamed nodes: <index>_<op_type>
    for idx, node in enumerate(graph.node):
        if not node.name:
            node.name = f"{idx}_{node.op_type}"

    # Collect existing output names so we don't duplicate
    existing_output_names = {o.name for o in graph.output}
    original_output_names = set(existing_output_names)

    # Collect all intermediate tensor names (node outputs)
    intermediate_names = []
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name not in existing_output_names:
                intermediate_names.append(output_name)

    # Try shape inference to get type info for intermediates
    try:
        model = shape_inference.infer_shapes(model)
        graph = model.graph
    except Exception as e:
        logger.warning("Shape inference failed (%s), proceeding with empty type info", e)

    # Build a lookup of known value_info by name
    known_vi = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        known_vi[vi.name] = vi

    # Add intermediate tensors as graph outputs
    for name in intermediate_names:
        if name in known_vi:
            graph.output.append(known_vi[name])
        else:
            # Empty type info — ORT can still return the tensor
            graph.output.append(helper.make_empty_tensor_value_info(name))

    return model, original_output_names


def build_initializer_table(model: ModelProto) -> dict[str, np.ndarray]:
    """Extract all graph initializers as a name -> numpy array dict.

    Args:
        model: An ONNX ModelProto.

    Returns:
        Dict mapping initializer tensor names to numpy arrays.
    """
    table = {}
    for init in model.graph.initializer:
        table[init.name] = numpy_helper.to_array(init)
    return table
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/onnx_dump/graph.py tests/conftest.py tests/test_graph.py
git commit -m "feat: graph surgery — load, validate, augment ONNX model"
```

---

## Chunk 2: Runner (Inference) and Exporter

### Task 3: Runner — input mapping and inference

**Files:**
- Create: `src/onnx_dump/runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for runner**

`tests/test_runner.py`:

```python
"""Tests for runner.py — input mapping and ORT inference."""

import numpy as np
import onnx
import pytest
from onnx_dump.graph import load_and_augment, build_initializer_table
from onnx_dump.runner import run_inference, map_inputs_from_files


class TestRunInference:
    def test_simple_add(self, simple_add_model):
        """Should return all tensors including the output Z."""
        augmented, _ = load_and_augment(simple_add_model)
        inputs = {
            "X": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            "Y": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32),
        }
        results = run_inference(augmented, inputs)
        assert "Z" in results
        np.testing.assert_allclose(results["Z"], inputs["X"] + inputs["Y"])

    def test_multi_op_all_intermediates(self, multi_op_model):
        """Should capture matmul_out, add_out, and output tensors."""
        augmented, _ = load_and_augment(multi_op_model)
        X = np.random.randn(2, 4).astype(np.float32)
        results = run_inference(augmented, {"X": X})
        assert "matmul_out" in results
        assert "add_out" in results
        assert "output" in results

    def test_multi_op_shapes(self, multi_op_model):
        """Intermediate tensor shapes should match expected dimensions."""
        augmented, _ = load_and_augment(multi_op_model)
        X = np.random.randn(2, 4).astype(np.float32)
        results = run_inference(augmented, {"X": X})
        assert results["matmul_out"].shape == (2, 3)
        assert results["add_out"].shape == (2, 3)
        assert results["output"].shape == (2, 3)


class TestMapInputsFromFiles:
    def test_map_by_declaration_order(self, tmp_path, simple_add_model):
        """Should map .npy files to model inputs by declaration order."""
        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))
        np.save(y_path, np.ones((2, 3), dtype=np.float32) * 2)

        result = map_inputs_from_files(simple_add_model, [x_path, y_path])
        assert "X" in result
        assert "Y" in result
        np.testing.assert_array_equal(result["X"], np.ones((2, 3)))

    def test_explicit_input_names(self, tmp_path, simple_add_model):
        """Should use explicit input_names when provided."""
        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))
        np.save(y_path, np.ones((2, 3), dtype=np.float32) * 2)

        result = map_inputs_from_files(
            simple_add_model, [x_path, y_path], input_names=["X", "Y"]
        )
        assert "X" in result and "Y" in result

    def test_input_names_length_mismatch(self, tmp_path, simple_add_model):
        """Should raise ValueError if input_names length != input_paths length."""
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="input_names length"):
            map_inputs_from_files(
                simple_add_model, [x_path], input_names=["X", "Y"]
            )

    def test_skips_initializer_backed_inputs(self, tmp_path, multi_op_model):
        """Should skip initializer-backed inputs when mapping by order."""
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.ones((2, 4), dtype=np.float32))

        init_names = {init.name for init in multi_op_model.graph.initializer}
        result = map_inputs_from_files(
            multi_op_model, [x_path], initializer_names=init_names
        )
        assert "X" in result
        assert len(result) == 1  # Only X, not W or B

    def test_count_mismatch_error(self, tmp_path, simple_add_model):
        """Should raise ValueError when too few inputs provided."""
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="Expected 2"):
            map_inputs_from_files(simple_add_model, [x_path])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'onnx_dump.runner'`

- [ ] **Step 3: Implement runner.py**

`src/onnx_dump/runner.py`:

```python
"""Inference runner — map inputs, run ORT session, collect all tensors."""

import logging

import numpy as np
import onnxruntime as ort
from onnx import ModelProto

logger = logging.getLogger("onnx_dump")


def run_inference(
    model: ModelProto,
    input_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run inference on the augmented model and return all output tensors.

    Args:
        model: Augmented ONNX ModelProto (with intermediate outputs exposed).
        input_arrays: Dict mapping input names to numpy arrays.

    Returns:
        Dict mapping tensor names to numpy arrays (all graph outputs).
    """
    # Serialize model to bytes for ORT
    model_bytes = model.SerializeToString()

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # suppress ORT warnings
    session = ort.InferenceSession(model_bytes, sess_options, providers=["CPUExecutionProvider"])

    # Validate that all required inputs are provided
    session_inputs = {inp.name for inp in session.get_inputs()}
    provided = set(input_arrays.keys())
    missing = session_inputs - provided
    if missing:
        raise ValueError(
            f"Missing inputs: {sorted(missing)}. "
            f"Model expects: {sorted(session_inputs)}. "
            f"Provided: {sorted(provided)}."
        )

    # Run — request all outputs
    output_names = [o.name for o in session.get_outputs()]
    results_list = session.run(output_names, input_arrays)

    results = dict(zip(output_names, results_list))
    logger.info("Inference complete: captured %d tensors", len(results))
    return results


def map_inputs_from_files(
    model: ModelProto,
    input_paths: list[str],
    input_names: list[str] | None = None,
    initializer_names: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load .npy files and map them to model input names.

    Args:
        model: ONNX ModelProto.
        input_paths: List of paths to .npy files.
        input_names: Optional explicit name mapping. If None, uses declaration order
                     (skipping initializer-backed inputs).
        initializer_names: Set of tensor names that have initializers.

    Returns:
        Dict mapping input names to numpy arrays.
    """
    if initializer_names is None:
        initializer_names = set()

    if input_names is not None:
        if len(input_names) != len(input_paths):
            raise ValueError(
                f"input_names length ({len(input_names)}) must match "
                f"input_paths length ({len(input_paths)})"
            )
        return {name: np.load(path) for name, path in zip(input_names, input_paths)}

    # Map by declaration order, skipping initializer-backed inputs
    model_input_names = [
        inp.name for inp in model.graph.input
        if inp.name not in initializer_names
    ]

    if len(input_paths) != len(model_input_names):
        raise ValueError(
            f"Expected {len(model_input_names)} input files for non-initializer inputs "
            f"{model_input_names}, got {len(input_paths)}"
        )

    return {name: np.load(path) for name, path in zip(model_input_names, input_paths)}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_runner.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/onnx_dump/runner.py tests/test_runner.py
git commit -m "feat: inference runner with input mapping"
```

---

### Task 4: Exporter — manifest.json and .npy files

**Files:**
- Create: `src/onnx_dump/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing tests for exporter**

`tests/test_exporter.py`:

```python
"""Tests for exporter.py — manifest.json generation and .npy file output."""

import json
import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_dump.exporter import export_results


class TestExportResults:
    def _make_simple_model(self):
        """Helper: X + Y = Z model."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        add = helper.make_node("Add", ["X", "Y"], ["Z"], name="Add_0")
        graph = helper.make_graph([add], "test", [X, Y], [Z])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_writes_manifest_json(self, tmp_path):
        """Should produce a valid manifest.json file."""
        model = self._make_simple_model()
        tensor_data = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
        }
        export_results(model, tensor_data, {}, str(tmp_path / "out"),
                       original_output_names={"Z"})

        manifest_path = tmp_path / "out" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == "1.0"
        assert len(manifest["nodes"]) == 1
        assert manifest["nodes"][0]["op_type"] == "Add"
        assert manifest["nodes"][0]["name"] == "Add_0"

    def test_writes_npy_files(self, tmp_path):
        """Should write .npy files for each tensor."""
        model = self._make_simple_model()
        tensor_data = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
        }
        out_dir = str(tmp_path / "out")
        export_results(model, tensor_data, {}, out_dir,
                       original_output_names={"Z"})

        for name in ["X", "Y", "Z"]:
            npy_path = tmp_path / "out" / "tensors" / f"{name}.npy"
            assert npy_path.exists()
            loaded = np.load(str(npy_path))
            np.testing.assert_array_equal(loaded, tensor_data[name])

    def test_manifest_tensor_metadata(self, tmp_path):
        """Each tensor entry should have name, shape, dtype, data_path."""
        model = self._make_simple_model()
        tensor_data = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
        }
        export_results(model, tensor_data, {}, str(tmp_path / "out"),
                       original_output_names={"Z"})

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        node = manifest["nodes"][0]
        for inp in node["inputs"]:
            assert "name" in inp
            assert "shape" in inp
            assert "dtype" in inp
            assert "data_path" in inp
            assert inp["data_path"].startswith("tensors/")

    def test_shared_tensor_written_once(self, tmp_path):
        """A tensor consumed by multiple nodes should produce one .npy file."""
        # Model: X -> Abs -> abs_out; X -> Neg -> neg_out
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        out1 = helper.make_tensor_value_info("abs_out", TensorProto.FLOAT, [2])
        out2 = helper.make_tensor_value_info("neg_out", TensorProto.FLOAT, [2])
        abs_n = helper.make_node("Abs", ["X"], ["abs_out"], name="Abs_0")
        neg_n = helper.make_node("Neg", ["X"], ["neg_out"], name="Neg_0")
        graph = helper.make_graph([abs_n, neg_n], "shared", [X], [out1, out2])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        tensor_data = {
            "X": np.array([1.0, -2.0], dtype=np.float32),
            "abs_out": np.array([1.0, 2.0], dtype=np.float32),
            "neg_out": np.array([-1.0, 2.0], dtype=np.float32),
        }
        export_results(model, tensor_data, {}, str(tmp_path / "out"),
                       original_output_names={"abs_out", "neg_out"})

        # X.npy should exist only once
        x_path = tmp_path / "out" / "tensors" / "X.npy"
        assert x_path.exists()

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        # Both nodes should reference the same data_path for X
        abs_node = manifest["nodes"][0]
        neg_node = manifest["nodes"][1]
        x_refs = [i["data_path"] for n in manifest["nodes"] for i in n["inputs"] if i["name"] == "X"]
        assert all(r == "tensors/X.npy" for r in x_refs)

    def test_initializer_included_in_export(self, tmp_path):
        """Initializer tensors should be exported as .npy and referenced in manifest."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        B_data = np.ones((3,), dtype=np.float32)
        B_init = helper.make_tensor("B", TensorProto.FLOAT, [3], B_data.tobytes())
        add = helper.make_node("Add", ["X", "B"], ["Z"], name="Add_0")
        graph = helper.make_graph([add], "with_init", [X], [Z], initializer=[B_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        initializer_table = {"B": B_data}
        tensor_data = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Z": np.ones((2, 3), dtype=np.float32) * 2,
        }
        export_results(model, tensor_data, initializer_table, str(tmp_path / "out"),
                       original_output_names={"Z"})

        b_path = tmp_path / "out" / "tensors" / "B.npy"
        assert b_path.exists()
        np.testing.assert_array_equal(np.load(str(b_path)), B_data)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_exporter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'onnx_dump.exporter'`

- [ ] **Step 3: Implement exporter.py**

`src/onnx_dump/exporter.py`:

```python
"""Exporter — write manifest.json and per-tensor .npy files."""

import json
import logging
import os
from pathlib import Path

import numpy as np
from onnx import ModelProto, AttributeProto, numpy_helper

logger = logging.getLogger("onnx_dump")

# 10 GB threshold for large model warning
_SIZE_WARNING_BYTES = 10 * 1024 * 1024 * 1024


def export_results(
    model: ModelProto,
    inference_results: dict[str, np.ndarray],
    initializer_table: dict[str, np.ndarray],
    output_dir: str,
    model_path: str = "",
    original_output_names: set[str] | None = None,
) -> None:
    """Write manifest.json and .npy tensor files to output_dir.

    Args:
        model: The augmented ONNX model (with named nodes).
        inference_results: Dict of tensor_name -> numpy array from inference.
        initializer_table: Dict of initializer tensor_name -> numpy array.
        output_dir: Directory to write output files.
        model_path: Original model file path (recorded in manifest).
        original_output_names: Set of output names from before graph surgery.
    """
    if original_output_names is None:
        original_output_names = {o.name for o in model.graph.output}

    out_path = Path(output_dir)
    tensors_path = out_path / "tensors"
    tensors_path.mkdir(parents=True, exist_ok=True)

    # Merge all tensor sources for lookup
    all_tensors: dict[str, np.ndarray] = {}
    all_tensors.update(initializer_table)
    all_tensors.update(inference_results)

    # Track written files to avoid duplicate writes and handle name collisions
    written_files: dict[str, str] = {}  # tensor_name -> relative data_path
    used_filenames: set[str] = set()  # track filenames to detect collisions
    total_bytes = 0

    def _write_tensor(name: str, node_name: str = "") -> str | None:
        """Write a tensor to .npy if not already written. Returns relative data_path."""
        nonlocal total_bytes
        if name in written_files:
            return written_files[name]
        arr = all_tensors.get(name)
        if arr is None:
            return None
        # Handle name collision: fall back to node_name__tensor_name
        filename = f"{name}.npy"
        if filename in used_filenames:
            filename = f"{node_name}__{name}.npy" if node_name else f"dup_{name}.npy"
        used_filenames.add(filename)
        data_path = f"tensors/{filename}"
        np.save(str(tensors_path / filename), arr)
        total_bytes += arr.nbytes
        written_files[name] = data_path
        return data_path

    def _tensor_entry(name: str) -> dict | None:
        """Build a manifest entry for a tensor."""
        data_path = _write_tensor(name)
        arr = all_tensors.get(name)
        if arr is None:
            return {"name": name, "shape": None, "dtype": None, "data_path": None}
        return {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data_path": data_path,
        }

    # Extract opset version for default domain
    opset_version = 0
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break

    # Build per-node manifest entries
    nodes = []
    for node in model.graph.node:
        inputs = [_tensor_entry(name) for name in node.input if name]
        outputs = [_tensor_entry(name) for name in node.output if name]
        # Write tensors referenced by this node
        for name in node.input:
            if name:
                _write_tensor(name, node.name)
        for name in node.output:
            if name:
                _write_tensor(name, node.name)
        attrs = _serialize_attributes(node, tensors_path)
        nodes.append({
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain,
            "attributes": attrs,
            "inputs": inputs,
            "outputs": outputs,
        })

    # Build graph-level I/O entries
    graph_inputs = [_tensor_entry(inp.name) for inp in model.graph.input
                    if inp.name not in initializer_table]
    # Only include ORIGINAL graph outputs, not augmented intermediates
    graph_outputs_manifest = [
        _tensor_entry(out_vi.name) for out_vi in model.graph.output
        if out_vi.name in original_output_names
    ]

    manifest = {
        "schema_version": "1.0",
        "model_path": model_path,
        "opset_version": opset_version,
        "nodes": nodes,
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs_manifest,
    }

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if total_bytes > _SIZE_WARNING_BYTES:
        logger.warning(
            "Total tensor size %.1f GB exceeds 10 GB threshold",
            total_bytes / (1024 ** 3),
        )

    logger.info(
        "Exported %d nodes, %d tensor files to %s",
        len(nodes), len(written_files), output_dir,
    )


def _serialize_attributes(node, tensors_path: Path) -> dict:
    """Serialize ONNX node attributes to JSON-compatible dict."""
    attrs = {}
    for attr in node.attribute:
        if attr.type == AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode("utf-8")
        elif attr.type == AttributeProto.FLOATS:
            attrs[attr.name] = list(attr.floats)
        elif attr.type == AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == AttributeProto.STRINGS:
            attrs[attr.name] = [s.decode("utf-8") for s in attr.strings]
        elif attr.type == AttributeProto.TENSOR:
            arr = numpy_helper.to_array(attr.t)
            filename = f"attr_{node.name}_{attr.name}.npy"
            np.save(str(tensors_path / filename), arr)
            attrs[attr.name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "data_path": f"tensors/{filename}",
            }
        elif attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
            attrs[attr.name] = None  # Graph attributes omitted
        else:
            attrs[attr.name] = str(attr)  # Fallback
    return attrs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_exporter.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/onnx_dump/exporter.py tests/test_exporter.py
git commit -m "feat: exporter — manifest.json and .npy tensor output"
```

---

## Chunk 3: Public API, CLI, and End-to-End Tests

### Task 5: Public API — dump_model()

**Files:**
- Modify: `src/onnx_dump/__init__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for dump_model()**

Add to `tests/test_cli.py`:

```python
"""Tests for the public API and CLI."""

import json
import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_dump import dump_model


class TestDumpModel:
    def test_end_to_end_with_path(self, tmp_path):
        """dump_model() should accept a model path and produce correct output."""
        # Build simple model: Z = X + Y
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        add = helper.make_node("Add", ["X", "Y"], ["Z"], name="Add_0")
        graph = helper.make_graph([add], "test", [X, Y], [Z])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)

        x_path = str(tmp_path / "x.npy")
        y_path = str(tmp_path / "y.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))
        np.save(y_path, np.ones((2, 3), dtype=np.float32) * 2)

        out_dir = str(tmp_path / "output")
        dump_model(model_path, [x_path, y_path], out_dir)

        manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
        assert manifest["schema_version"] == "1.0"
        assert manifest["model_path"] == model_path
        assert len(manifest["nodes"]) == 1

        z = np.load(str(tmp_path / "output" / "tensors" / "Z.npy"))
        np.testing.assert_allclose(z, np.ones((2, 3)) * 3)

    def test_with_input_names(self, tmp_path):
        """dump_model() should accept explicit input_names."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
        abs_n = helper.make_node("Abs", ["X"], ["Y"], name="Abs_0")
        graph = helper.make_graph([abs_n], "test", [X], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)

        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.array([-1.0, 2.0], dtype=np.float32))

        out_dir = str(tmp_path / "output")
        dump_model(model_path, [x_path], out_dir, input_names=["X"])

        y = np.load(str(tmp_path / "output" / "tensors" / "Y.npy"))
        np.testing.assert_allclose(y, np.array([1.0, 2.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL — `ImportError: cannot import name 'dump_model' from 'onnx_dump'`

- [ ] **Step 3: Implement dump_model() in __init__.py**

`src/onnx_dump/__init__.py`:

```python
"""ONNX Operator Dump Tool — capture per-operator intermediate tensors."""

__version__ = "0.1.0"

from onnx_dump.graph import load_and_augment, build_initializer_table
from onnx_dump.runner import run_inference, map_inputs_from_files
from onnx_dump.exporter import export_results


def dump_model(
    model_path: str,
    input_paths: list[str],
    output_dir: str = "./onnx_dump_output/",
    input_names: list[str] | None = None,
) -> None:
    """Run an ONNX model and dump per-operator intermediate tensors.

    Args:
        model_path: Path to .onnx model file.
        input_paths: List of paths to input .npy files.
        output_dir: Directory for output manifest.json and tensor .npy files.
        input_names: Optional explicit mapping of .npy files to model input names.
                     If None, maps by declaration order (skipping initializer-backed inputs).
    """
    model, original_output_names = load_and_augment(model_path)
    init_table = build_initializer_table(model)
    input_arrays = map_inputs_from_files(
        model, input_paths, input_names,
        initializer_names=set(init_table.keys()),
    )

    results = run_inference(model, input_arrays)

    # Add user inputs to inference results so exporter can reference them
    for name, arr in input_arrays.items():
        if name not in results:
            results[name] = arr

    export_results(
        model, results, init_table, output_dir,
        model_path=model_path,
        original_output_names=original_output_names,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/onnx_dump/__init__.py tests/test_cli.py
git commit -m "feat: dump_model() public API"
```

---

### Task 6: CLI wrapper

**Files:**
- Create: `src/onnx_dump/cli.py`
- Modify: `tests/test_cli.py` (add CLI tests)

- [ ] **Step 1: Write failing test for CLI**

Append to `tests/test_cli.py`:

```python
import subprocess
import sys


class TestCLI:
    def test_version_flag(self):
        """--version should print the version string."""
        result = subprocess.run(
            [sys.executable, "-m", "onnx_dump", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_cli_end_to_end(self, tmp_path):
        """CLI should produce output directory with manifest."""
        # Build model
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
        abs_n = helper.make_node("Abs", ["X"], ["Y"], name="Abs_0")
        graph = helper.make_graph([abs_n], "test", [X], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)

        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.array([-3.0, 4.0], dtype=np.float32))

        out_dir = str(tmp_path / "output")
        result = subprocess.run(
            [sys.executable, "-m", "onnx_dump", model_path, x_path, "-o", out_dir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert (tmp_path / "output" / "manifest.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py::TestCLI -v
```

Expected: FAIL — `No module named onnx_dump.__main__` or similar

- [ ] **Step 3: Implement cli.py and __main__.py**

`src/onnx_dump/cli.py`:

```python
"""CLI wrapper for onnx-dump."""

import argparse
import logging
import sys

from onnx_dump import __version__, dump_model


def main():
    parser = argparse.ArgumentParser(
        prog="onnx-dump",
        description="Dump per-operator intermediate tensors from an ONNX model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("model", help="Path to .onnx model file")
    parser.add_argument("inputs", nargs="+", help="Path(s) to input .npy files")
    parser.add_argument("-o", "--output-dir", default="./onnx_dump_output/",
                        help="Output directory (default: ./onnx_dump_output/)")
    parser.add_argument("--input-names", default=None,
                        help="Comma-separated input names to map .npy files to model inputs")

    args = parser.parse_args()

    # Set up logging for CLI
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_names = None
    if args.input_names:
        input_names = [n.strip() for n in args.input_names.split(",")]

    try:
        dump_model(args.model, args.inputs, args.output_dir, input_names=input_names)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

`src/onnx_dump/__main__.py`:

```python
"""Allow running as python -m onnx_dump."""

from onnx_dump.cli import main

main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/onnx_dump/cli.py src/onnx_dump/__main__.py tests/test_cli.py
git commit -m "feat: CLI wrapper with --version and argparse"
```

---

### Task 7: System test — end-to-end accuracy

**Files:**
- Create: `tests/test_system.py`

- [ ] **Step 1: Write the system test**

`tests/test_system.py`:

```python
"""System test — end-to-end accuracy verification.

Builds a multi-op model (MatMul -> Add -> Relu), computes expected outputs
manually with numpy, runs dump_model(), and verifies every dumped tensor
matches the numpy reference values.
"""

import json
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_dump import dump_model


class TestSystemAccuracy:
    def test_matmul_add_relu_accuracy(self, tmp_path):
        """Every intermediate tensor should match manually computed numpy values."""
        # Fixed seed for reproducibility
        rng = np.random.RandomState(42)

        # Model inputs and weights
        X_data = rng.randn(2, 4).astype(np.float32)
        W_data = rng.randn(4, 3).astype(np.float32)
        B_data = rng.randn(3).astype(np.float32)

        # Expected intermediate values (numpy reference)
        expected_matmul = X_data @ W_data
        expected_add = expected_matmul + B_data
        expected_relu = np.maximum(0, expected_add)

        # Build ONNX model: X -> MatMul(W) -> Add(B) -> Relu -> output
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

        W_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 3], W_data.flatten().tolist())
        B_init = helper.make_tensor("B", TensorProto.FLOAT, [3], B_data.flatten().tolist())

        matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="MatMul_0")
        add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"], name="Add_0")
        relu = helper.make_node("Relu", ["add_out"], ["output"], name="Relu_0")

        graph = helper.make_graph(
            [matmul, add, relu], "accuracy_test",
            [X], [out],
            initializer=[W_init, B_init],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)

        # Save model and input
        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, X_data)

        # Run dump_model
        out_dir = str(tmp_path / "output")
        dump_model(model_path, [x_path], out_dir)

        # Load manifest and verify structure
        manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
        assert len(manifest["nodes"]) == 3
        assert [n["op_type"] for n in manifest["nodes"]] == ["MatMul", "Add", "Relu"]

        # Load and verify each intermediate tensor
        tensors_dir = tmp_path / "output" / "tensors"

        matmul_out = np.load(str(tensors_dir / "matmul_out.npy"))
        np.testing.assert_allclose(matmul_out, expected_matmul, rtol=1e-6, atol=1e-6)

        add_out = np.load(str(tensors_dir / "add_out.npy"))
        np.testing.assert_allclose(add_out, expected_add, rtol=1e-6, atol=1e-6)

        output = np.load(str(tensors_dir / "output.npy"))
        np.testing.assert_allclose(output, expected_relu, rtol=1e-6, atol=1e-6)

        # Verify input tensor was also dumped
        x_dumped = np.load(str(tensors_dir / "X.npy"))
        np.testing.assert_array_equal(x_dumped, X_data)

        # Verify initializers were dumped
        w_dumped = np.load(str(tensors_dir / "W.npy"))
        np.testing.assert_array_equal(w_dumped, W_data)
        b_dumped = np.load(str(tensors_dir / "B.npy"))
        np.testing.assert_array_equal(b_dumped, B_data)
```

- [ ] **Step 2: Run system test**

```bash
uv run pytest tests/test_system.py -v
```

Expected: All 1 test PASS

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_system.py
git commit -m "test: end-to-end accuracy test for MatMul->Add->Relu"
```

---

### Task 8: README and examples

**Files:**
- Create: `README.md`
- Create: `examples/run_example.py`

- [ ] **Step 1: Write README.md**

```markdown
# onnx-dump

Dump per-operator intermediate tensors from ONNX models. Produces a `manifest.json` with operator metadata and `.npy` files for every tensor — useful as a golden reference for comparing chip-specific operator implementations.

## Install

```bash
uv sync
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
# Basic usage
python -m onnx_dump model.onnx input_x.npy -o ./output/

# With explicit input names
python -m onnx_dump model.onnx input_x.npy --input-names X -o ./output/

# Version
python -m onnx_dump --version
```

## Output

```
output/
├── manifest.json        # Per-operator metadata
└── tensors/
    ├── X.npy            # Input tensor
    ├── matmul_out.npy   # Intermediate tensor
    └── output.npy       # Final output
```

`manifest.json` contains operator details (name, op_type, attributes) with shape/dtype/path for every input and output tensor.

## Run tests

```bash
uv run pytest -v
```

## Example

```bash
python examples/run_example.py
```
```

- [ ] **Step 2: Write examples/run_example.py**

```python
"""Quick-start example: build a tiny model and dump its tensors."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_dump import dump_model


def main():
    # Build a simple model: Z = Relu(X + Y)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])

    add = helper.make_node("Add", ["X", "Y"], ["add_out"], name="Add_0")
    relu = helper.make_node("Relu", ["add_out"], ["Z"], name="Relu_0")

    graph = helper.make_graph([add, relu], "example", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    model_path = "/tmp/example_model.onnx"
    onnx.save(model, model_path)

    # Create inputs
    x_path = "/tmp/input_x.npy"
    y_path = "/tmp/input_y.npy"
    np.save(x_path, np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32))
    np.save(y_path, np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32))

    # Dump
    dump_model(model_path, [x_path, y_path], "/tmp/onnx_dump_example/")
    print("Done! Output in /tmp/onnx_dump_example/")
    print("  manifest.json — operator metadata")
    print("  tensors/      — .npy tensor files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add README.md examples/run_example.py
git commit -m "docs: README and quick-start example"
```

---

## Chunk 4: Final Verification

### Task 9: Full test suite and cleanup

- [ ] **Step 1: Run complete test suite**

```bash
uv run pytest -v --tb=short
```

Expected: All tests PASS

- [ ] **Step 2: Verify CLI works end-to-end**

```bash
uv run python examples/run_example.py
ls /tmp/onnx_dump_example/
cat /tmp/onnx_dump_example/manifest.json | python -m json.tool
```

Expected: manifest.json displays properly with 2 nodes (Add_0, Relu_0), tensor files exist.

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "chore: final cleanup"
```
