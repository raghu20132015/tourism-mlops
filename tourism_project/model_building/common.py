"""Shared helpers used by every pipeline step: configuration, paths, file I/O and cleaning."""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../tourism_project

# Columns from the data dictionary and the type each one should have.
EXPECTED_SCHEMA = {
    "CustomerID": "numeric",
    "ProdTaken": "numeric",
    "Age": "numeric",
    "TypeofContact": "categorical",
    "CityTier": "numeric",
    "Occupation": "categorical",
    "Gender": "categorical",
    "NumberOfPersonVisiting": "numeric",
    "PreferredPropertyStar": "numeric",
    "MaritalStatus": "categorical",
    "NumberOfTrips": "numeric",
    "Passport": "numeric",
    "OwnCar": "numeric",
    "NumberOfChildrenVisiting": "numeric",
    "Designation": "categorical",
    "MonthlyIncome": "numeric",
    "PitchSatisfactionScore": "numeric",
    "ProductPitched": "categorical",
    "NumberOfFollowups": "numeric",
    "DurationOfPitch": "numeric",
}
CATEGORICAL = [c for c, t in EXPECTED_SCHEMA.items() if t == "categorical"]


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(PROJECT_DIR / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--data", default=None, help="Override data.raw_path from the config")
    return parser.parse_args()


def load_config(args):
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.data:
        cfg["data"]["raw_path"] = args.data
    return cfg


def resolve(path):
    """Resolve a config path relative to tourism_project/."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def read_raw(path):
    """Read the dataset straight from the repository data folder; drop pandas index columns."""
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Add Tourism.csv to tourism_project/data/.")
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])


def clean_frame(df, category_map):
    """Trim text, fix category spellings and coerce numeric columns. Missing values stay NaN."""
    df = df.copy()
    for col in df.columns:
        if col in CATEGORICAL:
            stripped = df[col].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = stripped.replace(category_map.get(col, {}))
        elif EXPECTED_SCHEMA.get(col) == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_json(obj, path):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def load_json(path):
    with open(resolve(path), encoding="utf-8") as fh:
        return json.load(fh)
