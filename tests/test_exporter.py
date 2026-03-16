"""Tests for exporter.py - manifest.json generation and .npy file output."""

import json

import numpy as np
from onnx import TensorProto, helper

from onnx_dump.exporter import export_results


class TestExportResults:
    def _make_simple_model(self):
        """Helper: X + Y = Z model."""
        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        y_input = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        z_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        add = helper.make_node("Add", ["X", "Y"], ["Z"], name="Add_0")
        graph = helper.make_graph([add], "test", [x_input, y_input], [z_output])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    def test_writes_manifest_json(self, tmp_path):
        """Should produce a valid manifest.json file."""
        model = self._make_simple_model()
        tensor_data = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
        }

        export_results(model, tensor_data, {}, str(tmp_path / "out"), original_output_names={"Z"})

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

        export_results(model, tensor_data, {}, out_dir, original_output_names={"Z"})

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

        export_results(model, tensor_data, {}, str(tmp_path / "out"), original_output_names={"Z"})

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        node = manifest["nodes"][0]
        for model_input in node["inputs"]:
            assert "name" in model_input
            assert "shape" in model_input
            assert "dtype" in model_input
            assert "data_path" in model_input
            assert model_input["data_path"].startswith("tensors/")

    def test_shared_tensor_written_once(self, tmp_path):
        """A tensor consumed by multiple nodes should produce one .npy file."""
        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
        out1 = helper.make_tensor_value_info("abs_out", TensorProto.FLOAT, [2])
        out2 = helper.make_tensor_value_info("neg_out", TensorProto.FLOAT, [2])
        abs_node = helper.make_node("Abs", ["X"], ["abs_out"], name="Abs_0")
        neg_node = helper.make_node("Neg", ["X"], ["neg_out"], name="Neg_0")
        graph = helper.make_graph([abs_node, neg_node], "shared", [x_input], [out1, out2])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        tensor_data = {
            "X": np.array([1.0, -2.0], dtype=np.float32),
            "abs_out": np.array([1.0, 2.0], dtype=np.float32),
            "neg_out": np.array([-1.0, 2.0], dtype=np.float32),
        }

        export_results(
            model,
            tensor_data,
            {},
            str(tmp_path / "out"),
            original_output_names={"abs_out", "neg_out"},
        )

        x_path = tmp_path / "out" / "tensors" / "X.npy"
        assert x_path.exists()

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        x_refs = [
            model_input["data_path"]
            for node in manifest["nodes"]
            for model_input in node["inputs"]
            if model_input["name"] == "X"
        ]
        assert all(ref == "tensors/X.npy" for ref in x_refs)

    def test_initializer_included_in_export(self, tmp_path):
        """Initializer tensors should be exported as .npy and referenced in manifest."""
        x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        z_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        b_data = np.ones((3,), dtype=np.float32)
        b_init = helper.make_tensor(
            "B",
            TensorProto.FLOAT,
            [3],
            b_data.astype(np.float32).tolist(),
        )
        add = helper.make_node("Add", ["X", "B"], ["Z"], name="Add_0")
        graph = helper.make_graph([add], "with_init", [x_input], [z_output], initializer=[b_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        initializer_table = {"B": b_data}
        tensor_data = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Z": np.ones((2, 3), dtype=np.float32) * 2,
        }

        export_results(
            model,
            tensor_data,
            initializer_table,
            str(tmp_path / "out"),
            original_output_names={"Z"},
        )

        b_path = tmp_path / "out" / "tensors" / "B.npy"
        assert b_path.exists()
        np.testing.assert_array_equal(np.load(str(b_path)), b_data)
