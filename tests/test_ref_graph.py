"""Tests for the reference graph builder."""

from onnx import helper, TensorProto

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

    def test_default_opset_selected_before_custom_domains(self):
        """Default-domain opset should drive meta even if the import list shuffles entries."""
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="Add_0")
        graph = helper.make_graph([add_node], "mixed_opset", [], [])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("com.example", 5),
                helper.make_opsetid("", 19),
            ],
        )
        initializer_table = build_initializer_table(model)

        result = build_ref_graph(model, inference_results={}, initializer_table=initializer_table)

        assert result["meta"]["opset_version"] == 19

    def test_attribute_normalization_filters_unsupported_types(self):
        """Attributes of unsupported kinds should collapse to None."""
        tensor_attr = helper.make_tensor("invalid", TensorProto.FLOAT, [1], [1.0])
        attr_invalid = helper.make_attribute("invalid", tensor_attr)
        attr_message = helper.make_attribute("message", "ok")
        node = helper.make_node(
            "Relu",
            inputs=["X"],
            outputs=["Y"],
            name="Relu_0",
        )
        node.attribute.extend([attr_invalid, attr_message])
        graph = helper.make_graph([node], "attr_filter", [], [])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        initializer_table = build_initializer_table(model)
        result = build_ref_graph(model, inference_results={}, initializer_table=initializer_table)

        step = result["steps"][0]
        assert step["attributes"]["invalid"] is None
        assert step["attributes"]["message"] == "ok"
