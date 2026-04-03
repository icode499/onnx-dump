"""Reference graph builder helpers."""

from __future__ import annotations

from typing import Any

from onnx import AttributeProto, helper
from onnx import ModelProto


def _get_default_opset(model: ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain == "":
            return int(opset.version)
    domains = [op.domain or "" for op in model.opset_import]
    raise ValueError(
        f"Default-domain ONNX opset import missing; present imports: {domains}"
    )

_ALLOWED_ATTRIBUTE_TYPES = {
    AttributeProto.FLOAT,
    AttributeProto.INT,
    AttributeProto.STRING,
    AttributeProto.FLOATS,
    AttributeProto.INTS,
    AttributeProto.STRINGS,
}


def _normalize_attribute(attribute: AttributeProto) -> Any:
    if attribute.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS}:
        return None
    if attribute.type not in _ALLOWED_ATTRIBUTE_TYPES:
        return None
    value = helper.get_attribute_value(attribute)
    if attribute.type == AttributeProto.STRING and isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    if attribute.type == AttributeProto.STRINGS:
        return [item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else item for item in value]
    return value


def build_ref_graph(model: ModelProto, inference_results: dict[str, Any], initializer_table: dict[str, Any]) -> dict[str, Any]:
    """Build the reference JSON schema for the given ONNX model."""

    meta = {
        "format_version": 1,
        "graph_spec": "onnx",
        "opset_version": _get_default_opset(model),
    }

    steps: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        node_name = node.name or f"{index}_{node.op_type}"
        attributes = {
            attribute.name: _normalize_attribute(attribute)
            for attribute in node.attribute
        }

        steps.append(
            {
                "id": node_name,
                "name": node_name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": attributes,
            }
        )

    return {
        "meta": meta,
        "steps": steps,
        "tensors": {},
    }
