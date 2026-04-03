"""Tests for the reference graph builder."""

import numpy as np
from onnx import helper, TensorProto

import pytest

from onnx_dump.graph import build_initializer_table, load_and_augment

from onnx_dump.ref_graph import build_ref_graph


class TestBuildRefGraph:
    def test_emits_meta_and_steps(self, simple_add_model):
        """Should emit the expected meta block and a single add step."""
        model, _ = load_and_augment(simple_add_model)
        initializer_table = build_initializer_table(model)

        inference_results = {
            "X": np.zeros((2, 3), dtype=np.float32),
            "Y": np.zeros((2, 3), dtype=np.float32),
            "Z": np.zeros((2, 3), dtype=np.float32),
        }

        result = build_ref_graph(
            model, inference_results=inference_results, initializer_table=initializer_table
        )

        meta = result["meta"]
        assert meta["format_version"] == 1
        assert meta["graph_spec"] == "onnx"
        expected_opset = next(import_.version for import_ in model.opset_import if import_.domain == "")
        assert meta["opset_version"] == expected_opset

        steps = result["steps"]
        assert len(steps) == 1
        step = steps[0]
        assert step["id"] == "Add_0"
        assert step["name"] == "Add_0"
        assert step["op_type"] == "Add"
        assert step["inputs"] == ["X", "Y"]
        assert step["outputs"] == ["Z"]
        assert step["attributes"] == {}

        assert isinstance(result["tensors"], dict)

    def test_raises_when_referenced_tensor_missing_from_all_sources(self, simple_add_model):
        model, _ = load_and_augment(simple_add_model)
        initializer_table = build_initializer_table(model)

        # Y is a required graph input for the Add node, but omitted here to
        # ensure the builder fails clearly when referenced tensors are missing.
        inference_results = {
            "X": np.zeros((2, 3), dtype=np.float32),
            "Z": np.zeros((2, 3), dtype=np.float32),
        }

        with pytest.raises(
            ValueError,
            match="referenced by the graph is missing from inference_results and initializer_table",
        ):
            build_ref_graph(
                model,
                inference_results=inference_results,
                initializer_table=initializer_table,
            )

    def test_emits_tensor_metadata_for_inputs_outputs_and_initializers(
        self, multi_op_model
    ):
        model, _ = load_and_augment(multi_op_model)
        initializer_table = build_initializer_table(model)

        inference_results = {
            "X": np.ones((2, 4), dtype=np.float32),
            "matmul_out": np.full((2, 3), 2.0, dtype=np.float32),
            "add_out": np.full((2, 3), 3.0, dtype=np.float32),
            "output": np.full((2, 3), 4.0, dtype=np.float32),
        }

        result = build_ref_graph(
            model,
            inference_results=inference_results,
            initializer_table=initializer_table,
        )

        tensor_table = result["tensors"]
        expected_tensor_names = {"X", "W", "B", "matmul_out", "add_out", "output"}
        assert set(tensor_table) == expected_tensor_names

        combined_arrays = {**initializer_table, **inference_results}
        for name in expected_tensor_names:
            entry = tensor_table[name]
            assert entry["dtype"] == "FLOAT"
            assert entry["storage_format"] == "plain"
            assert entry["shape"] == list(combined_arrays[name].shape)

    def test_uses_runtime_array_shape_as_source_of_truth(self, simple_add_model):
        model, _ = load_and_augment(simple_add_model)
        initializer_table = build_initializer_table(model)

        inference_results = {
            "X": np.zeros((4, 5), dtype=np.float32),
            "Y": np.ones((4, 5), dtype=np.float32),
            "Z": np.full((4, 5), 7.0, dtype=np.float32),
        }

        result = build_ref_graph(
            model,
            inference_results=inference_results,
            initializer_table=initializer_table,
        )

        tensor_table = result["tensors"]
        assert tensor_table["Z"]["shape"] == [4, 5]
        assert tensor_table["Z"]["dtype"] == "FLOAT"
        assert tensor_table["Z"]["storage_format"] == "plain"

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

        inference_results = {
            "X": np.zeros((1,), dtype=np.float32),
            "Y": np.zeros((1,), dtype=np.float32),
            "Z": np.zeros((1,), dtype=np.float32),
        }

        result = build_ref_graph(
            model, inference_results=inference_results, initializer_table=initializer_table
        )

        assert result["meta"]["opset_version"] == 19

    def test_ai_onnx_opset_is_accepted_as_default_fallback(self):
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="Add_0")
        graph = helper.make_graph([add_node], "ai_onnx_only", [], [])
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("ai.onnx", 18)],
        )

        inference_results = {
            "X": np.zeros((1,), dtype=np.float32),
            "Y": np.zeros((1,), dtype=np.float32),
            "Z": np.zeros((1,), dtype=np.float32),
        }

        result = build_ref_graph(model, inference_results=inference_results, initializer_table={})

        assert result["meta"]["opset_version"] == 18

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
        inference_results = {
            "X": np.zeros((1,), dtype=np.float32),
            "Y": np.zeros((1,), dtype=np.float32),
        }

        result = build_ref_graph(
            model, inference_results=inference_results, initializer_table=initializer_table
        )

        step = result["steps"][0]
        assert step["attributes"]["invalid"] is None
        assert step["attributes"]["message"] == "ok"

    @pytest.mark.parametrize("bad_name", ["bad/name", "bad\\name", ".", ".."])
    def test_rejects_unsafe_tensor_names_before_manifest_construction(self, bad_name):
        """Unsafe tensor names (e.g. path-like names) must be rejected."""
        add_node = helper.make_node(
            "Add",
            inputs=["X", "Y"],
            outputs=[bad_name],
            name="Add_0",
        )
        graph = helper.make_graph(
            [add_node],
            "unsafe_tensor_names",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1]),
            ],
            outputs=[helper.make_tensor_value_info(bad_name, TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        initializer_table = build_initializer_table(model)

        inference_results = {
            "X": np.zeros((1,), dtype=np.float32),
            "Y": np.zeros((1,), dtype=np.float32),
            bad_name: np.zeros((1,), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="Unsafe tensor name"):
            build_ref_graph(
                model,
                inference_results=inference_results,
                initializer_table=initializer_table,
            )

    def test_unnamed_nodes_get_auto_ids(self, unnamed_nodes_model):
        """Builder should synthesize names for nodes that start unnamed."""
        initializer_table = build_initializer_table(unnamed_nodes_model)

        inference_results = {
            "X": np.zeros((2, 3), dtype=np.float32),
            "abs_out": np.zeros((2, 3), dtype=np.float32),
            "output": np.zeros((2, 3), dtype=np.float32),
        }

        result = build_ref_graph(
            unnamed_nodes_model,
            inference_results=inference_results,
            initializer_table=initializer_table,
        )

        ids = [step["id"] for step in result["steps"]]
        expected_ids = [f"{index}_{node.op_type}" for index, node in enumerate(unnamed_nodes_model.graph.node)]
        assert ids == expected_ids

    def test_duplicate_node_names_get_unique_step_ids(self):
        first = helper.make_node("Abs", ["X"], ["abs_out"], name="dup")
        second = helper.make_node("Neg", ["abs_out"], ["Y"], name="dup")
        graph = helper.make_graph(
            [first, second],
            "duplicate_names",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        inference_results = {
            "X": np.zeros((1,), dtype=np.float32),
            "abs_out": np.zeros((1,), dtype=np.float32),
            "Y": np.zeros((1,), dtype=np.float32),
        }

        result = build_ref_graph(model, inference_results=inference_results, initializer_table={})

        assert [step["id"] for step in result["steps"]] == ["dup", "dup_1"]
        assert [step["name"] for step in result["steps"]] == ["dup", "dup"]

    def test_filters_blank_optional_inputs_and_outputs_from_steps(self):
        node = helper.make_node(
            "Dropout",
            inputs=["X", "", ""],
            outputs=["Y", ""],
            name="Dropout_0",
        )
        graph = helper.make_graph(
            [node],
            "optional_blanks",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        inference_results = {
            "X": np.zeros((1,), dtype=np.float32),
            "Y": np.zeros((1,), dtype=np.float32),
        }

        result = build_ref_graph(model, inference_results=inference_results, initializer_table={})

        assert result["steps"][0]["inputs"] == ["X"]
        assert result["steps"][0]["outputs"] == ["Y"]

    def test_missing_default_opset_raises(self):
        """Missing default-domain opset import must raise a clear error."""
        graph = helper.make_graph([], "bad", [], [])
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("com.example", 1)],
        )
        initializer_table = build_initializer_table(model)

        with pytest.raises(ValueError, match="Default-domain ONNX opset import missing"):
            build_ref_graph(model, inference_results={}, initializer_table=initializer_table)
