"""Tests for exporter.py - ref manifest and .npy file output."""

import copy
import json

import numpy as np
import pytest

from onnx_dump.exporter import export_results


class TestExportResults:
    def _graph_document(self) -> dict:
        return {
            "meta": {
                "format_version": 1,
                "graph_spec": "onnx",
                "opset_version": 17,
            },
            "steps": [
                {
                    "id": "Add_0",
                    "name": "Add_0",
                    "op_type": "Add",
                    "inputs": ["X", "Y"],
                    "outputs": ["Z"],
                    "attributes": {},
                }
            ],
            "tensors": {
                "X": {"dtype": "FLOAT", "shape": [2, 3], "storage_format": "plain"},
                "Y": {"dtype": "FLOAT", "shape": [2, 3], "storage_format": "plain"},
                "Z": {"dtype": "FLOAT", "shape": [2, 3], "storage_format": "plain"},
            },
        }

    def test_writes_ref_manifest_json(self, tmp_path):
        document = self._graph_document()
        original = copy.deepcopy(document)
        tensor_table = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
        }

        export_results(document, tensor_table, str(tmp_path / "out"))

        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest == original
        assert document == original
        assert "schema_version" not in manifest
        assert "nodes" not in manifest
        assert "graph_inputs" not in manifest
        assert "graph_outputs" not in manifest

    def test_writes_npy_files_for_declared_tensors(self, tmp_path):
        document = self._graph_document()
        tensor_table = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
            "EXTRA": np.array([42.0], dtype=np.float32),
        }

        export_results(document, tensor_table, str(tmp_path / "out"))

        written_files = sorted(path.name for path in (tmp_path / "out" / "tensors").glob("*.npy"))
        assert written_files == ["X.npy", "Y.npy", "Z.npy"]

        for name in ["X", "Y", "Z"]:
            npy_path = tmp_path / "out" / "tensors" / f"{name}.npy"
            assert npy_path.exists()
            np.testing.assert_array_equal(np.load(npy_path), tensor_table[name])

    def test_initializer_style_tensor_is_exported_when_declared(self, tmp_path):
        document = self._graph_document()
        document["steps"][0]["inputs"] = ["X", "B"]
        document["tensors"]["B"] = {
            "dtype": "FLOAT",
            "shape": [3],
            "storage_format": "plain",
        }

        tensor_table = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32) * 2,
            "Z": np.ones((2, 3), dtype=np.float32) * 3,
            "B": np.ones((3,), dtype=np.float32),
        }

        export_results(document, tensor_table, str(tmp_path / "out"))

        b_path = tmp_path / "out" / "tensors" / "B.npy"
        assert b_path.exists()
        np.testing.assert_array_equal(np.load(b_path), tensor_table["B"])

    def test_raises_when_declared_tensor_missing_from_tensor_table(self, tmp_path):
        document = self._graph_document()
        tensor_table = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="Tensor 'Z' missing from tensor_table"):
            export_results(document, tensor_table, str(tmp_path / "out"))

    @pytest.mark.parametrize("bad_name", ["bad/name", r"bad\\name", ".", ".."])
    def test_raises_when_declared_tensor_name_is_unsafe(self, tmp_path, bad_name):
        document = self._graph_document()
        document["tensors"][bad_name] = document["tensors"].pop("Z")
        document["steps"][0]["outputs"] = [bad_name]
        tensor_table = {
            "X": np.ones((2, 3), dtype=np.float32),
            "Y": np.ones((2, 3), dtype=np.float32),
            bad_name: np.ones((2, 3), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="Unsafe tensor name"):
            export_results(document, tensor_table, str(tmp_path / "out"))
