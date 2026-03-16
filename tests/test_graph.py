"""Tests for graph.py - model loading, validation, graph surgery, initializer extraction."""

import numpy as np
import onnx

from onnx_dump.graph import build_initializer_table, load_and_augment


class TestLoadAndAugment:
    def test_adds_intermediate_outputs(self, simple_add_model):
        """Graph surgery should add intermediate tensor Z as an output."""
        augmented, original_outputs = load_and_augment(simple_add_model)
        output_names = [output.name for output in augmented.graph.output]

        assert "Z" in output_names
        assert original_outputs == {"Z"}

    def test_multi_op_intermediates(self, multi_op_model):
        """All intermediate tensors should be in outputs."""
        augmented, original_outputs = load_and_augment(multi_op_model)
        output_names = [output.name for output in augmented.graph.output]

        assert "matmul_out" in output_names
        assert "add_out" in output_names
        assert "output" in output_names
        assert original_outputs == {"output"}

    def test_auto_names_unnamed_nodes(self, unnamed_nodes_model):
        """Unnamed nodes should get <index>_<op_type> names."""
        augmented, _ = load_and_augment(unnamed_nodes_model)
        node_names = [node.name for node in augmented.graph.node]

        assert node_names == ["0_Abs", "1_Neg"]

    def test_load_from_path(self, tmp_path, simple_add_model):
        """Should accept a file path string."""
        model_path = str(tmp_path / "model.onnx")
        onnx.save(simple_add_model, model_path)

        augmented, _ = load_and_augment(model_path)
        output_names = [output.name for output in augmented.graph.output]

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
