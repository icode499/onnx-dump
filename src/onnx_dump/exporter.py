"""Exporter - write manifest.json and per-tensor .npy files."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from onnx import AttributeProto, ModelProto, numpy_helper

logger = logging.getLogger("onnx_dump")

_SIZE_WARNING_BYTES = 10 * 1024 * 1024 * 1024


def export_results(
    model: ModelProto,
    inference_results: dict[str, np.ndarray],
    initializer_table: dict[str, np.ndarray],
    output_dir: str,
    model_path: str = "",
    original_output_names: set[str] | None = None,
) -> None:
    """Write manifest.json and .npy tensor files to output_dir."""
    if original_output_names is None:
        original_output_names = {output.name for output in model.graph.output}

    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    tensors_path = out_path / "tensors"
    tensors_path.mkdir(parents=True, exist_ok=True)

    all_tensors: dict[str, np.ndarray] = {}
    all_tensors.update(initializer_table)
    all_tensors.update(inference_results)

    written_files: dict[str, str] = {}
    used_filenames: set[str] = set()
    total_bytes = 0

    def _write_tensor(name: str, node_name: str = "") -> str | None:
        nonlocal total_bytes

        if name in written_files:
            return written_files[name]

        array = all_tensors.get(name)
        if array is None:
            return None

        filename = f"{name}.npy"
        if filename in used_filenames:
            filename = f"{node_name}__{name}.npy" if node_name else f"dup_{name}.npy"

        used_filenames.add(filename)
        data_path = f"tensors/{filename}"
        np.save(str(tensors_path / filename), array)
        total_bytes += array.nbytes
        written_files[name] = data_path
        return data_path

    def _tensor_entry(name: str, node_name: str = "") -> dict:
        data_path = _write_tensor(name, node_name=node_name)
        array = all_tensors.get(name)
        if array is None:
            return {"name": name, "shape": None, "dtype": None, "data_path": None}

        return {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "data_path": data_path,
        }

    opset_version = 0
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset_version = opset.version
            break

    nodes = []
    for node in model.graph.node:
        inputs = [_tensor_entry(name, node.name) for name in node.input if name]
        outputs = [_tensor_entry(name, node.name) for name in node.output if name]
        attributes = _serialize_attributes(node, tensors_path)
        nodes.append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "domain": node.domain,
                "attributes": attributes,
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    graph_inputs = [
        _tensor_entry(model_input.name)
        for model_input in model.graph.input
        if model_input.name not in initializer_table
    ]
    graph_outputs = [
        _tensor_entry(output.name)
        for output in model.graph.output
        if output.name in original_output_names
    ]

    manifest = {
        "schema_version": "1.0",
        "model_path": model_path,
        "opset_version": opset_version,
        "nodes": nodes,
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
    }

    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if total_bytes > _SIZE_WARNING_BYTES:
        logger.warning(
            "Total tensor size %.1f GB exceeds 10 GB threshold",
            total_bytes / (1024 ** 3),
        )

    logger.info(
        "Exported %d nodes, %d tensor files to %s",
        len(nodes),
        len(written_files),
        output_dir,
    )


def _serialize_attributes(node, tensors_path: Path) -> dict:
    """Serialize ONNX node attributes to JSON-compatible values."""
    attributes = {}
    for attr in node.attribute:
        if attr.type == AttributeProto.FLOAT:
            attributes[attr.name] = attr.f
        elif attr.type == AttributeProto.INT:
            attributes[attr.name] = attr.i
        elif attr.type == AttributeProto.STRING:
            attributes[attr.name] = attr.s.decode("utf-8")
        elif attr.type == AttributeProto.FLOATS:
            attributes[attr.name] = list(attr.floats)
        elif attr.type == AttributeProto.INTS:
            attributes[attr.name] = list(attr.ints)
        elif attr.type == AttributeProto.STRINGS:
            attributes[attr.name] = [value.decode("utf-8") for value in attr.strings]
        elif attr.type == AttributeProto.TENSOR:
            array = numpy_helper.to_array(attr.t)
            filename = f"attr_{node.name}_{attr.name}.npy"
            np.save(str(tensors_path / filename), array)
            attributes[attr.name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "data_path": f"tensors/{filename}",
            }
        elif attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
            attributes[attr.name] = None
        else:
            attributes[attr.name] = str(attr)
    return attributes
