#!/usr/bin/env python3
"""Inspect a parquet file: show its schema (fields + types), row count,
and print the first N rows.

Usage:
    python inspect_parquet.py <path_to_parquet> [num_rows]

Example:
    python inspect_parquet.py data/train.parquet
    python inspect_parquet.py data/train.parquet 3
"""
import sys

import pyarrow.parquet as pq


def inspect(path: str, num_rows: int = 3) -> None:
    pf = pq.ParquetFile(path)

    # --- row count (read from metadata, no need to load the whole file) ---
    total_rows = pf.metadata.num_rows
    num_cols = pf.metadata.num_columns

    print("=" * 70)
    print(f"File: {path}")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {num_cols}")
    print(f"Row groups: {pf.metadata.num_row_groups}")

    # --- fields / schema ---
    print("-" * 70)
    print("Fields (name : type):")
    for field in pf.schema_arrow:
        print(f"  {field.name} : {field.type}")

    # --- first N rows ---
    print("-" * 70)
    n = min(num_rows, total_rows)
    print(f"First {n} row(s):")

    # Read only the first batch large enough to cover num_rows, then slice.
    head = next(pf.iter_batches(batch_size=max(n, 1))).slice(0, n)
    table = head

    try:
        import pandas as pd  # optional, nicer display

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 120)
        print(table.to_pandas().to_string(index=True))
    except ImportError:
        # Fall back to plain dict-per-row output if pandas isn't installed.
        rows = table.to_pylist()
        for i, row in enumerate(rows):
            print(f"\n[row {i}]")
            for k, v in row.items():
                v_str = repr(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + f"... (len={len(v_str)})"
                print(f"  {k}: {v_str}")

    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    inspect(path, num_rows)
