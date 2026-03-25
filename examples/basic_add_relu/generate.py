"""Generate a self-contained Add -> Relu example under this directory."""

from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper

from onnx_dump import dump_model

EXAMPLE_ROOT = Path(__file__).resolve().parent


def _build_model() -> ModelProto:
    """Build the deterministic Add -> Relu example model."""
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y_input = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    z_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])

    add = helper.make_node("Add", ["X", "Y"], ["add_out"], name="Add_0")
    relu = helper.make_node("Relu", ["add_out"], ["Z"], name="Relu_0")

    graph = helper.make_graph([add, relu], "basic_add_relu", [x_input, y_input], [z_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def _build_inputs() -> dict[str, np.ndarray]:
    """Return deterministic example inputs keyed by graph input name."""
    return {
        "X": np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32),
        "Y": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32),
    }


def _reset_directory(path: Path) -> None:
    """Remove and recreate a directory to keep generated artifacts in sync."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def generate_example(base_dir: Path | None = None) -> Path:
    """Generate the named example dataset under base_dir."""
    example_dir = EXAMPLE_ROOT if base_dir is None else Path(base_dir)

    input_dir = example_dir / "input"
    input_tensors_dir = input_dir / "tensors"
    output_dir = example_dir / "output"

    _reset_directory(input_dir)
    _reset_directory(output_dir)
    input_tensors_dir.mkdir(parents=True, exist_ok=True)

    model_path = input_dir / "model.onnx"
    onnx.save(_build_model(), model_path)

    input_paths = []
    for name, array in _build_inputs().items():
        tensor_path = input_tensors_dir / f"{name}.npy"
        np.save(tensor_path, array)
        input_paths.append(str(tensor_path))

    dump_model(str(model_path), input_paths, str(output_dir))

    print(f"Generated example in {example_dir}")
    print("  input/model.onnx")
    print("  input/tensors/*.npy")
    print("  output/manifest.json")
    print("  output/tensors/*.npy")
    return example_dir


def main() -> None:
    """Generate the example in place."""
    generate_example()


if __name__ == "__main__":
    main()
