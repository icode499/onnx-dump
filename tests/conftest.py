"""Shared test fixtures - programmatic ONNX model builders."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def simple_add_model():
    """Model: Z = X + Y (two inputs, one Add node, one output)."""
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y_input = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    z_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])

    add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="Add_0")

    graph = helper.make_graph([add_node], "simple_add", [x_input, y_input], [z_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def multi_op_model():
    """Model: X -> MatMul(W) -> Add(B) -> Relu -> output."""
    rng = np.random.RandomState(123)
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    w_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [4, 3],
        rng.randn(4, 3).astype(np.float32).flatten().tolist(),
    )
    b_init = helper.make_tensor(
        "B",
        TensorProto.FLOAT,
        [3],
        rng.randn(3).astype(np.float32).flatten().tolist(),
    )

    matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="MatMul_0")
    add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"], name="Add_0")
    relu = helper.make_node("Relu", ["add_out"], ["output"], name="Relu_0")

    graph = helper.make_graph(
        [matmul, add, relu],
        "multi_op",
        [x_input],
        [output],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def unnamed_nodes_model():
    """Model with unnamed nodes: X -> Abs -> Neg -> output."""
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])

    abs_node = helper.make_node("Abs", ["X"], ["abs_out"])
    neg_node = helper.make_node("Neg", ["abs_out"], ["output"])

    graph = helper.make_graph([abs_node, neg_node], "unnamed", [x_input], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model
