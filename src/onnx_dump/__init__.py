def hello() -> str:
    return "Hello from onnx-dump!"
"""ONNX Operator Dump Tool - capture per-operator intermediate tensors."""

from onnx_dump.exporter import export_results
from onnx_dump.graph import build_initializer_table, load_and_augment
from onnx_dump.runner import map_inputs_from_files, run_inference

__version__ = "0.1.0"


def dump_model(
    model_path: str,
    input_paths: list[str],
    output_dir: str = "./onnx_dump_output/",
    input_names: list[str] | None = None,
) -> None:
    """Run an ONNX model and dump per-operator intermediate tensors."""
    model, original_output_names = load_and_augment(model_path)
    initializer_table = build_initializer_table(model)
    input_arrays = map_inputs_from_files(
        model,
        input_paths,
        input_names,
        initializer_names=set(initializer_table.keys()),
    )

    results = run_inference(model, input_arrays)
    for name, array in input_arrays.items():
        if name not in results:
            results[name] = array

    export_results(
        model,
        results,
        initializer_table,
        output_dir,
        model_path=model_path,
        original_output_names=original_output_names,
    )
