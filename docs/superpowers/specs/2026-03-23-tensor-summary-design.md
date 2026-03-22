# Tensor Summary & Preview in manifest.json

**Date:** 2026-03-23
**Status:** Draft

## Problem

The `onnx-dump` tool outputs all tensor data as `.npy` files. Users cannot see basic tensor information (statistics, values) without writing extra code to load and inspect them. This makes the tool unfriendly for quick debugging and comparison workflows.

## Solution

Enhance the existing `manifest.json` with per-tensor summary statistics and a data preview. Additionally, print a human-readable summary table to the terminal after dump completes.

## Design

### 1. manifest.json Changes

**Schema version:** `"1.0"` → `"1.1"` (additive change only — new `summary` field is added to tensor entries. Consumers of `1.0` manifests SHOULD ignore unknown fields.)

Each tensor entry in `inputs`, `outputs`, `graph_inputs`, and `graph_outputs` gains a `summary` field:

```json
{
  "name": "Add_0__output",
  "shape": [1, 3, 224, 224],
  "dtype": "float32",
  "data_path": "tensors/Add_0__output.npy",
  "summary": {
    "min": -2.345,
    "max": 8.901,
    "mean": 1.234,
    "std": 2.567,
    "zeros": 128,
    "nan_count": 0,
    "inf_count": 0,
    "num_elements": 150528,
    "preview": "[0.123, -1.456, 2.789, 0.0, 3.142, 0.567, -2.345, 1.890, ...] (150528 elements)"
  }
}
```

**Field definitions:**

| Field | Type | Description |
|-------|------|-------------|
| `min` | float | Minimum value (NaN-aware) |
| `max` | float | Maximum value (NaN-aware) |
| `mean` | float | Arithmetic mean |
| `std` | float | Standard deviation |
| `zeros` | int | Number of elements equal to zero |
| `nan_count` | int | Number of NaN elements |
| `inf_count` | int | Number of Inf/-Inf elements |
| `num_elements` | int | Total element count (`array.size`) |
| `preview` | string | Flattened first 8 elements as string, with `...` and total count if truncated |

**Numeric dtypes** (float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64): receive all statistical fields. Note that `mean` and `std` of integer arrays produce float results.

**Non-numeric dtypes** (bool, string, object): only `num_elements` and `preview` are populated; statistical fields (`min`, `max`, `mean`, `std`, `zeros`, `nan_count`, `inf_count`) are omitted.

**Empty tensors** (0 elements): `summary` contains only `num_elements: 0` and `preview: "[]"`.

**JSON serialization of special float values:** JSON has no `NaN`/`Infinity` literals. When a stat value is NaN (e.g., `nanmin` on an all-NaN array) or Inf, serialize it as `null`. This ensures valid JSON output. Example: an all-NaN float32 tensor produces `"min": null, "max": null, "mean": null, "std": null`.

**Missing tensors:** When a tensor name is not found in the results dict (i.e., `_tensor_entry` produces `data_path: null`), no `summary` field is added.

**Preview format:**
- If `num_elements <= 8`: `"[1.0, 2.0, 3.0]"` (all elements, no ellipsis)
- If `num_elements > 8`: `"[0.123, -1.456, 2.789, 0.0, 3.142, 0.567, -2.345, 1.890, ...] (150528 elements)"`
- Values are formatted using Python's `f"{x:.6g}"` format specifier (6 significant digits, scientific notation for very large/small values, e.g., `1.23457e+06`).

### 2. Terminal Summary Table

After export completes, print a summary table via `logger.info`.

**Column widths and truncation:** Tensor names longer than 22 characters are truncated with `...` (e.g., `"/model/layer4/laye..."`). Shape strings longer than 15 characters are similarly truncated.

```
Tensor Summary (5 tensors, 1.2 MB total):
Name                  Shape           Dtype    Min       Max       Mean      Std
──────────────────────────────────────────────────────────────────────────────────
X                     [1,3,224,224]   float32  -2.345    8.901     1.234     2.567
weight                [64,3,3,3]      float32  -0.123    0.456     0.001     0.089
Add_0__output         [1,64,112,112]  float32   0.000   12.345     3.456     1.234
Relu_0__output        [1,64,112,112]  float32   0.000   12.345     4.567     1.123
output                [1,1000]        float32  -5.678    9.012     0.001     1.234
```

If any tensor contains NaN or Inf, append a warning line:
```
WARNING: 2 tensors contain NaN/Inf values: tensor_a, tensor_b
```

### 3. Implementation Approach

All changes are in `src/onnx_dump/exporter.py` — no new files.

**New function: `_compute_summary(array: np.ndarray) -> dict`**
- Takes a numpy array, returns a dict with summary fields
- Handles edge cases: empty arrays, non-numeric dtypes, NaN/Inf values
- Uses `np.nanmin`, `np.nanmax`, `np.nanmean`, `np.nanstd` for NaN-safe stats

**Modified: `_build_tensor_entry()` (or equivalent tensor metadata builder)**
- Calls `_compute_summary()` and adds the result as `"summary"` to each tensor dict

**Modified: `export_results()`**
- After writing all files and manifest, collects summary data and prints the table
- Updates `schema_version` to `"1.1"`

**New function: `_print_summary_table(nodes, ...)`**
- Formats and prints the terminal table using string formatting (no external deps)

### 4. Testing

Add tests in `tests/test_exporter.py`:

- **test_summary_stats_correct**: verify min/max/mean/std against known arrays
- **test_summary_with_nan_inf**: array containing NaN and Inf values, verify `null` in JSON
- **test_summary_all_nan**: all-NaN array, verify all stats are `null`
- **test_summary_empty_array**: 0-element tensor
- **test_summary_bool_dtype**: boolean tensor (stats omitted)
- **test_summary_integer_dtype**: integer tensor receives full stats
- **test_preview_truncation**: verify `...` for arrays > 8 elements
- **test_preview_small_array**: verify no `...` for arrays <= 8 elements
- **test_schema_version_updated**: manifest has `schema_version: "1.1"`
- **test_print_summary_table_format**: capture logger output and verify table structure
- **test_summary_table_nan_warning**: verify warning line when tensors contain NaN/Inf

### 5. Non-Goals

- No new output file formats (CSV, HTML, etc.)
- No new CLI subcommands
- No new dependencies
- No changes to the `.npy` file writing itself
