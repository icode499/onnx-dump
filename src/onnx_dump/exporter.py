"""Exporter - write ref-side manifest.json and per-tensor .npy files."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("onnx_dump")

_SIZE_WARNING_BYTES = 10 * 1024 * 1024 * 1024


def export_results(
    graph_document: dict[str, Any],
    tensor_table: dict[str, np.ndarray],
    output_dir: str,
) -> None:
    """Write manifest.json and declared tensor .npy files to output_dir."""
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    tensors_path = out_path / "tensors"
    tensors_path.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    tensor_entries = graph_document["tensors"]

    for tensor_name in sorted(tensor_entries):
        array = tensor_table.get(tensor_name)
        if array is None:
            raise ValueError(f"Tensor {tensor_name!r} missing from tensor_table")

        array = np.asarray(array)
        np.save(tensors_path / f"{tensor_name}.npy", array)
        total_bytes += array.nbytes

    (out_path / "manifest.json").write_text(json.dumps(graph_document, indent=2))

    if total_bytes > _SIZE_WARNING_BYTES:
        logger.warning(
            "Total tensor size %.1f GB exceeds 10 GB threshold",
            total_bytes / (1024 ** 3),
        )

    logger.info(
        "Exported %d tensor files to %s",
        len(tensor_entries),
        output_dir,
    )
