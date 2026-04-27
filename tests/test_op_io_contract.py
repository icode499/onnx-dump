from pathlib import Path

from op_io_contracts import validate_unified_graph


def test_basic_add_relu_output_satisfies_shared_ref_graph_contract():
    output_dir = Path("examples/basic_add_relu/output")

    result = validate_unified_graph(
        output_dir / "manifest.json",
        tensor_dir=output_dir / "tensors",
        role="ref",
    )

    assert result.ok, result.errors
