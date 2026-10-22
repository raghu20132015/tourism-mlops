"""Step 1 - Register and validate Tourism.csv.

Usage: python tourism_project/model_building/data_registration.py [--data <path>]
"""
import sys

import pandas as pd

from common import EXPECTED_SCHEMA, CATEGORICAL, load_config, parse_args, read_raw, resolve, save_json


def validate(df, cfg):
    """Return (errors, warnings). Errors stop the pipeline; warnings are handled by cleaning."""
    errors, warnings = [], []
    target = cfg["target"]

    missing_cols = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")
    if extra_cols:
        warnings.append(f"Columns not in the data dictionary (ignored): {extra_cols}")

    for col, kind in EXPECTED_SCHEMA.items():
        if col in df.columns and kind == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            errors.append(f"'{col}' should be numeric; {int(bad.sum())} values cannot be converted")

    if target in df.columns:
        if df[target].isna().any():
            errors.append(f"Target '{target}' has {int(df[target].isna().sum())} missing values")
        values = set(df[target].dropna().unique())
        if not values <= {0, 1}:
            errors.append(f"Target '{target}' must be 0/1, found {sorted(values)}")

    for id_col in cfg["id_columns"]:
        if id_col in df.columns and df[id_col].duplicated().any():
            errors.append(f"'{id_col}' has {int(df[id_col].duplicated().sum())} duplicate values")

    dup_rows = df.drop(columns=[c for c in cfg["id_columns"] if c in df.columns]).duplicated().sum()
    if dup_rows:
        warnings.append(f"{int(dup_rows)} rows are identical apart from the ID (removed during preparation)")

    for col, mapping in cfg["cleaning"]["category_map"].items():
        if col in df.columns:
            found = sorted(set(mapping) & set(df[col].dropna().astype(str).str.strip()))
            if found:
                warnings.append(f"'{col}' contains {found}; cleaning maps them to {[mapping[v] for v in found]}")
    return errors, warnings


def build_report(df, cfg, errors, warnings):
    target = cfg["target"]
    missing = df.isna().sum()
    return {
        "source": str(resolve(cfg["data"]["raw_path"])),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in missing.items() if n > 0},
        "category_values": {c: df[c].value_counts(dropna=False).to_dict() for c in CATEGORICAL if c in df.columns},
        "target_balance": df[target].value_counts().to_dict() if target in df.columns else {},
        "errors": errors,
        "warnings": warnings,
    }


def print_report(report):
    print(f"=== Data registration: {report['source']} ===")
    print(f"Shape: {report['rows']} rows x {report['columns']} columns\n")
    print(f"{'Column':<26} {'Type':<9} Expected")
    for col, dtype in report["dtypes"].items():
        print(f"{col:<26} {dtype:<9} {EXPECTED_SCHEMA.get(col, '-')}")
    print("\nMissing values:", report["missing_values"] or "none")
    print("Target balance:", report["target_balance"])
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for e in report["errors"]:
        print(f"ERROR: {e}")


def main():
    args = parse_args("Validate the Tourism dataset")
    cfg = load_config(args)
    try:
        df = read_raw(cfg["data"]["raw_path"])
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    errors, warnings = validate(df, cfg)
    report = build_report(df, cfg, errors, warnings)
    print_report(report)
    save_json(report, f"{cfg['paths']['artifacts_dir']}/data_report.json")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s).")
        sys.exit(1)
    print("\nValidation passed. Dataset registered.")


if __name__ == "__main__":
    main()
