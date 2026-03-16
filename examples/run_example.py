"""Quick-start example: build a tiny model and dump its tensors."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_dump import dump_model


def main() -> None:
    """Build a small model, run the dump, and print output locations."""
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y_input = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    z_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])

    add = helper.make_node("Add", ["X", "Y"], ["add_out"], name="Add_0")
    relu = helper.make_node("Relu", ["add_out"], ["Z"], name="Relu_0")

    graph = helper.make_graph([add, relu], "example", [x_input, y_input], [z_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    model_path = "/tmp/example_model.onnx"
    onnx.save(model, model_path)

    x_path = "/tmp/input_x.npy"
    y_path = "/tmp/input_y.npy"
    np.save(x_path, np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32))
    np.save(y_path, np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32))

    dump_model(model_path, [x_path, y_path], "/tmp/onnx_dump_example/")
    print("Done! Output in /tmp/onnx_dump_example/")
    print("  manifest.json - operator metadata")
    print("  tensors/      - .npy tensor files")


if __name__ == "__main__":
    main()
