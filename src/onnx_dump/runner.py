"""Inference runner - map inputs, run ORT session, collect all tensors."""

import logging

import numpy as np
import onnxruntime as ort
from onnx import ModelProto

logger = logging.getLogger("onnx_dump")


def run_inference(
    model: ModelProto,
    input_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run inference on the augmented model and return all output tensors."""
    model_bytes = model.SerializeToString()

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        model_bytes,
        session_options,
        providers=["CPUExecutionProvider"],
    )

    session_inputs = {session_input.name for session_input in session.get_inputs()}
    provided = set(input_arrays.keys())
    missing = session_inputs - provided
    if missing:
        raise ValueError(
            f"Missing inputs: {sorted(missing)}. "
            f"Model expects: {sorted(session_inputs)}. "
            f"Provided: {sorted(provided)}."
        )

    output_names = [output.name for output in session.get_outputs()]
    result_list = session.run(output_names, input_arrays)

    results = dict(zip(output_names, result_list))
    logger.info("Inference complete: captured %d tensors", len(results))
    return results


def map_inputs_from_files(
    model: ModelProto,
    input_paths: list[str],
    input_names: list[str] | None = None,
    initializer_names: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load .npy files and map them to model input names."""
    if initializer_names is None:
        initializer_names = set()

    if input_names is not None:
        if len(input_names) != len(input_paths):
            raise ValueError(
                f"input_names length ({len(input_names)}) must match "
                f"input_paths length ({len(input_paths)})"
            )
        return {
            name: np.load(path)
            for name, path in zip(input_names, input_paths)
        }

    model_input_names = [
        model_input.name
        for model_input in model.graph.input
        if model_input.name not in initializer_names
    ]
    if len(input_paths) != len(model_input_names):
        raise ValueError(
            f"Expected {len(model_input_names)} input files for non-initializer inputs "
            f"{model_input_names}, got {len(input_paths)}"
        )

    return {
        name: np.load(path)
        for name, path in zip(model_input_names, input_paths)
    }
