"""System test - end-to-end accuracy verification."""

import json

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_dump import dump_model


class TestSystemAccuracy:
    def test_matmul_add_relu_accuracy(self, tmp_path):
        """Every intermediate tensor should match manually computed numpy values."""
        rng = np.random.RandomState(42)

        x_data = rng.randn(2, 4).astype(np.float32)
        w_data = rng.randn(4, 3).astype(np.float32)
        b_data = rng.randn(3).astype(np.float32)

        expected_matmul = x_data @ w_data
        expected_add = expected_matmul + b_data
        expected_relu = np.maximum(0, expected_add)

        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

        w_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 3], w_data.flatten().tolist())
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [3], b_data.flatten().tolist())

        matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="MatMul_0")
        add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"], name="Add_0")
        relu = helper.make_node("Relu", ["add_out"], ["output"], name="Relu_0")

        graph = helper.make_graph(
            [matmul, add, relu],
            "accuracy_test",
            [x_input],
            [output],
            initializer=[w_init, b_init],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)

        model_path = str(tmp_path / "model.onnx")
        onnx.save(model, model_path)
        x_path = str(tmp_path / "x.npy")
        np.save(x_path, x_data)

        out_dir = str(tmp_path / "output")
        dump_model(model_path, [x_path], out_dir)

        manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
        assert manifest["meta"] == {
            "format_version": 1,
            "graph_spec": "onnx",
            "opset_version": 17,
        }
        assert [step["op_type"] for step in manifest["steps"]] == ["MatMul", "Add", "Relu"]
        assert manifest["tensors"]["output"]["dtype"] == "FLOAT"
        assert manifest["tensors"]["W"]["storage_format"] == "plain"

        tensors_dir = tmp_path / "output" / "tensors"

        matmul_out = np.load(str(tensors_dir / "matmul_out.npy"))
        np.testing.assert_allclose(matmul_out, expected_matmul, rtol=1e-6, atol=1e-6)

        add_out = np.load(str(tensors_dir / "add_out.npy"))
        np.testing.assert_allclose(add_out, expected_add, rtol=1e-6, atol=1e-6)

        output_tensor = np.load(str(tensors_dir / "output.npy"))
        np.testing.assert_allclose(output_tensor, expected_relu, rtol=1e-6, atol=1e-6)

        x_dumped = np.load(str(tensors_dir / "X.npy"))
        np.testing.assert_array_equal(x_dumped, x_data)

        w_dumped = np.load(str(tensors_dir / "W.npy"))
        np.testing.assert_array_equal(w_dumped, w_data)
        b_dumped = np.load(str(tensors_dir / "B.npy"))
        np.testing.assert_array_equal(b_dumped, b_data)
