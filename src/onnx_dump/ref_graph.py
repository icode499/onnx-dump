"""Reference graph builder helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import AttributeProto, helper
from onnx import ModelProto


def _get_default_opset(model: ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain == "":
            return int(opset.version)
    for opset in model.opset_import:
        if opset.domain == "ai.onnx":
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

_NUMPY_DTYPE_TO_ONNX = {
    "float16": "FLOAT16",
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "bool": "BOOL",
    "bool_": "BOOL",
    "bfloat16": "BFLOAT16",
    "complex64": "COMPLEX64",
    "complex128": "COMPLEX128",
    "float8_e4m3fn": "FLOAT8E4M3FN",
    "float8_e4m3fnuz": "FLOAT8E4M3FNUZ",
    "float8_e5m2": "FLOAT8E5M2",
    "float8_e5m2fnuz": "FLOAT8E5M2FNUZ",
}


def _numpy_dtype_to_onnx(dtype: np.dtype) -> str:
    dtype_name = np.dtype(dtype).name
    if dtype_name not in _NUMPY_DTYPE_TO_ONNX:
        raise ValueError(f"Unsupported numpy dtype {dtype_name!r}")
    return _NUMPY_DTYPE_TO_ONNX[dtype_name]


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


def _validate_tensor_name(name: str) -> None:
    # These names will later be used as keys and often as file-ish identifiers by
    # downstream tooling; reject path-like names early at the schema layer.
    if name in {".", ".."}:
        raise ValueError(f"Unsafe tensor name {name!r}: reserved path segment")
    if "/" in name or "\\" in name:
        raise ValueError(f"Unsafe tensor name {name!r}: contains a path separator")


def _unique_step_id(step_name: str, seen_ids: dict[str, int]) -> str:
    count = seen_ids.get(step_name, 0)
    seen_ids[step_name] = count + 1
    if count == 0:
        return step_name
    return f"{step_name}_{count}"


def build_ref_graph(model: ModelProto, inference_results: dict[str, Any], initializer_table: dict[str, Any]) -> dict[str, Any]:
    """Build the reference JSON schema for the given ONNX model.

    Note: `inference_results` is expected to contain NumPy arrays for graph
    inputs as well as node outputs (including intermediates) as produced by a
    runtime dump or equivalent mechanism.
    """

    meta = {
        "format_version": 1,
        "graph_spec": "onnx",
        "opset_version": _get_default_opset(model),
    }

    steps: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        node_name = node.name or f"{index}_{node.op_type}"
        step_id = _unique_step_id(node_name, seen_ids)
        attributes = {
            attribute.name: _normalize_attribute(attribute)
            for attribute in node.attribute
        }

        steps.append(
            {
                "id": step_id,
                "name": node_name,
                "op_type": node.op_type,
                "inputs": [name for name in node.input if name],
                "outputs": [name for name in node.output if name],
                "attributes": attributes,
            }
        )

    referenced_tensors: set[str] = set()
    for step in steps:
        referenced_tensors.update(name for name in step["inputs"] if name)
        referenced_tensors.update(name for name in step["outputs"] if name)

    # Ensure graph-level inputs/outputs are represented even if the step list
    # does not mention them (e.g., odd models, empty graphs).
    referenced_tensors.update(value_info.name for value_info in model.graph.input if value_info.name)
    referenced_tensors.update(value_info.name for value_info in model.graph.output if value_info.name)

    tensor_entries: dict[str, dict[str, Any]] = {}
    for name in sorted(referenced_tensors):
        _validate_tensor_name(name)
        array = inference_results.get(name)
        if array is None:
            array = initializer_table.get(name)
        if array is None:
            raise ValueError(
                f"Tensor {name!r} referenced by the graph is missing from inference_results and initializer_table"
            )

        array = np.asarray(array)
        tensor_entries[name] = {
            "dtype": _numpy_dtype_to_onnx(array.dtype),
            "shape": list(array.shape),
            "storage_format": "plain",
        }

    return {
        "meta": meta,
        "steps": steps,
        "tensors": tensor_entries,
    }
