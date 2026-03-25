"""Regression tests for repository examples."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


def load_example_module():
    """Load the basic_add_relu example generator module from disk."""
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "basic_add_relu"
        / "generate.py"
    )
    assert script_path.exists()

    spec = spec_from_file_location("basic_add_relu_generate", script_path)
    assert spec is not None
    assert spec.loader is not None

    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original

    return module


def test_basic_add_relu_generator_writes_model_to_named_layout(tmp_path):
    """The named example should generate input/model.onnx under its own layout."""
    module = load_example_module()

    example_dir = module.generate_example(tmp_path / "basic_add_relu")

    assert example_dir == tmp_path / "basic_add_relu"
    assert (example_dir / "input" / "model.onnx").exists()


def test_basic_add_relu_generator_writes_complete_example_data(tmp_path):
    """The named example should write aligned input and output tensor artifacts."""
    module = load_example_module()

    example_dir = module.generate_example(tmp_path / "basic_add_relu")

    input_tensors_dir = example_dir / "input" / "tensors"
    output_tensors_dir = example_dir / "output" / "tensors"

    x_input = np.load(input_tensors_dir / "X.npy")
    y_input = np.load(input_tensors_dir / "Y.npy")
    add_out = np.load(output_tensors_dir / "add_out.npy")
    z_output = np.load(output_tensors_dir / "Z.npy")

    np.testing.assert_array_equal(add_out, x_input + y_input)
    np.testing.assert_array_equal(z_output, np.maximum(0, add_out))

    manifest = json.loads((example_dir / "output" / "manifest.json").read_text())
    assert manifest["model_path"] == str(example_dir / "input" / "model.onnx")
    assert sorted(path.name for path in output_tensors_dir.glob("*.npy")) == [
        "X.npy",
        "Y.npy",
        "Z.npy",
        "add_out.npy",
    ]


def test_legacy_root_example_script_is_removed():
    """The repository should not keep the old top-level example entry point."""
    legacy_script = Path(__file__).resolve().parents[1] / "examples" / "run_example.py"
    assert not legacy_script.exists()
