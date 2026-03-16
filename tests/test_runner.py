"""Tests for runner.py - input mapping and ORT inference."""

import numpy as np
import pytest

from onnx_dump.graph import load_and_augment
from onnx_dump.runner import map_inputs_from_files, run_inference


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
        x_value = np.random.randn(2, 4).astype(np.float32)

        results = run_inference(augmented, {"X": x_value})

        assert "matmul_out" in results
        assert "add_out" in results
        assert "output" in results

    def test_multi_op_shapes(self, multi_op_model):
        """Intermediate tensor shapes should match expected dimensions."""
        augmented, _ = load_and_augment(multi_op_model)
        x_value = np.random.randn(2, 4).astype(np.float32)

        results = run_inference(augmented, {"X": x_value})

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
            simple_add_model,
            [x_path, y_path],
            input_names=["X", "Y"],
        )

        assert "X" in result and "Y" in result

    def test_input_names_length_mismatch(self, tmp_path, simple_add_model):
        """Should raise ValueError if input_names length != input_paths length."""
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="input_names length"):
            map_inputs_from_files(simple_add_model, [x_path], input_names=["X", "Y"])

    def test_skips_initializer_backed_inputs(self, tmp_path, multi_op_model):
        """Should skip initializer-backed inputs when mapping by order."""
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.ones((2, 4), dtype=np.float32))

        initializer_names = {initializer.name for initializer in multi_op_model.graph.initializer}
        result = map_inputs_from_files(
            multi_op_model,
            [x_path],
            initializer_names=initializer_names,
        )

        assert "X" in result
        assert len(result) == 1

    def test_count_mismatch_error(self, tmp_path, simple_add_model):
        """Should raise ValueError when too few inputs provided."""
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.ones((2, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="Expected 2"):
            map_inputs_from_files(simple_add_model, [x_path])
