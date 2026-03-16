"""CLI wrapper for onnx-dump."""

import argparse
import logging
import sys

from onnx_dump import __version__, dump_model


def main() -> None:
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="onnx-dump",
        description="Dump per-operator intermediate tensors from an ONNX model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("model", help="Path to .onnx model file")
    parser.add_argument("inputs", nargs="+", help="Path(s) to input .npy files")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./onnx_dump_output/",
        help="Output directory (default: ./onnx_dump_output/)",
    )
    parser.add_argument(
        "--input-names",
        default=None,
        help="Comma-separated input names to map .npy files to model inputs",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_names = None
    if args.input_names:
        input_names = [name.strip() for name in args.input_names.split(",")]

    try:
        dump_model(args.model, args.inputs, args.output_dir, input_names=input_names)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
