"""Graph surgery - load, validate, and augment ONNX models."""

import logging
from typing import Union

import numpy as np
import onnx
from onnx import ModelProto, helper, numpy_helper, shape_inference

logger = logging.getLogger("onnx_dump")


def load_and_augment(model_or_path: Union[str, ModelProto]) -> tuple[ModelProto, set[str]]:
    """Load, validate, and augment an ONNX model with intermediate outputs."""
    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path)
    else:
        model = model_or_path

    onnx.checker.check_model(model)

    graph = model.graph
    for index, node in enumerate(graph.node):
        if not node.name:
            node.name = f"{index}_{node.op_type}"

    existing_output_names = {output.name for output in graph.output}
    original_output_names = set(existing_output_names)
    intermediate_names: list[str] = []
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name not in existing_output_names:
                intermediate_names.append(output_name)

    try:
        model = shape_inference.infer_shapes(model)
        graph = model.graph
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Shape inference failed (%s), proceeding with empty type info",
            exc,
        )

    known_value_info = {}
    for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
        known_value_info[value_info.name] = value_info

    for name in intermediate_names:
        if name in known_value_info:
            graph.output.append(known_value_info[name])
        else:
            graph.output.append(helper.make_empty_tensor_value_info(name))

    return model, original_output_names


def build_initializer_table(model: ModelProto) -> dict[str, np.ndarray]:
    """Extract graph initializers as a name-to-array mapping."""
    table = {}
    for initializer in model.graph.initializer:
        table[initializer.name] = numpy_helper.to_array(initializer)
    return table
