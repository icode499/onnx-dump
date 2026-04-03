"""Tests for the reference graph builder."""

from onnx_dump.graph import build_initializer_table, load_and_augment

from onnx_dump.ref_graph import build_ref_graph


class TestBuildRefGraph:
    def test_emits_meta_and_steps(self, simple_add_model):
        """Should emit the expected meta block and a single add step."""
        model, _ = load_and_augment(simple_add_model)
        initializer_table = build_initializer_table(model)

        result = build_ref_graph(model, inference_results={}, initializer_table=initializer_table)

        meta = result["meta"]
        assert meta["format_version"] == 1
        assert meta["graph_spec"] == "onnx"
        assert meta["opset_version"] == 17

        steps = result["steps"]
        assert len(steps) == 1
        step = steps[0]
        assert step["id"] == "Add_0"
        assert step["name"] == "Add_0"
        assert step["op_type"] == "Add"
        assert step["inputs"] == ["X", "Y"]
        assert step["outputs"] == ["Z"]
        assert step["attributes"] == {}

        assert result["tensors"] == {}
