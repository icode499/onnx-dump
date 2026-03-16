"""Tests for the public API and CLI."""

import json
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_dump import dump_model


class TestDumpModel:
    def test_end_to_end_with_path(self, tmp_path):
        """dump_model() should accept a model path and produce correct output."""
        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        y_input = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        z_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        add = helper.make_node("Add", ["X", "Y"], ["Z"], name="Add_0")
        graph = helper.make_graph([add], "test", [x_input, y_input], [z_output])
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

        z_tensor = np.load(str(tmp_path / "output" / "tensors" / "Z.npy"))
        np.testing.assert_allclose(z_tensor, np.ones((2, 3)) * 3)

    def test_with_input_names(self, tmp_path):
        """dump_model() should accept explicit input_names."""
        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
        abs_node = helper.make_node("Abs", ["X"], ["Y"], name="Abs_0")
        graph = helper.make_graph([abs_node], "test", [x_input], [y_output])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)

        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.array([-1.0, 2.0], dtype=np.float32))

        out_dir = str(tmp_path / "output")
        dump_model(model_path, [x_path], out_dir, input_names=["X"])

        y_tensor = np.load(str(tmp_path / "output" / "tensors" / "Y.npy"))
        np.testing.assert_allclose(y_tensor, np.array([1.0, 2.0]))


class TestCLI:
    def test_version_flag(self):
        """--version should print the version string."""
        result = subprocess.run(
            [sys.executable, "-m", "onnx_dump", "--version"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_cli_end_to_end(self, tmp_path):
        """CLI should produce output directory with manifest."""
        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
        abs_node = helper.make_node("Abs", ["X"], ["Y"], name="Abs_0")
        graph = helper.make_graph([abs_node], "test", [x_input], [y_output])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)

        x_path = str(tmp_path / "x.npy")
        np.save(x_path, np.array([-3.0, 4.0], dtype=np.float32))

        out_dir = str(tmp_path / "output")
        result = subprocess.run(
            [sys.executable, "-m", "onnx_dump", model_path, x_path, "-o", out_dir],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert (tmp_path / "output" / "manifest.json").exists()
