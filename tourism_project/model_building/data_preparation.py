"""Step 2 - Clean Tourism.csv, split it into train/test, and record the feature schema.

Usage: python tourism_project/model_building/data_preparation.py [--data <path>]
"""
from sklearn.model_selection import train_test_split

from common import EXPECTED_SCHEMA, clean_frame, load_config, parse_args, read_raw, resolve, save_json


def build_schema(train, cfg, features):
    """Describe each feature from the training data (used by the app's input widgets)."""
    columns = {}
    for col in features:
        series = train[col].dropna()
        if EXPECTED_SCHEMA[col] == "categorical":
            columns[col] = {
                "type": "categorical",
                "values": sorted(series.astype(str).unique().tolist()),
                "default": str(series.mode().iloc[0]),
            }
        else:
            is_int = bool((series % 1 == 0).all())
            cast = int if is_int else float
            columns[col] = {
                "type": "numeric",
                "min": cast(series.min()),
                "max": cast(series.max()),
                "default": cast(round(series.median())) if is_int else float(series.median()),
                "integer": is_int,
                "binary": is_int and set(series.unique()) <= {0, 1},
            }
    return {
        "target": cfg["target"],
        "features": features,
        "post_contact_columns": cfg["features"]["post_contact_columns"],
        "category_map": cfg["cleaning"]["category_map"],
        "columns": columns,
    }


def main():
    args = parse_args("Clean and split the Tourism dataset")
    cfg = load_config(args)
    target = cfg["target"]

    # 1. Load directly from the repository data folder (index column dropped on read).
    df = read_raw(cfg["data"]["raw_path"])
    print(f"Loaded {len(df)} rows from {resolve(cfg['data']['raw_path'])}")

    # 2. Remove unnecessary columns: IDs and anything outside the data dictionary.
    drop_cols = [c for c in df.columns if c in cfg["id_columns"] or c not in EXPECTED_SCHEMA]
    df = df.drop(columns=drop_cols)
    print(f"Removed columns: {drop_cols}")

    # 3. Clean values, drop rows without a target, remove duplicates.
    df = clean_frame(df, cfg["cleaning"]["category_map"])
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    if cfg["cleaning"]["drop_duplicate_rows"]:
        before = len(df)
        df = df.drop_duplicates()
        print(f"Removed {before - len(df)} duplicate rows")
    print(f"Clean dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # 4. Stratified train/test split, saved locally.
    features = [c for c in df.columns if c != target]
    train, test = train_test_split(
        df, test_size=cfg["split"]["test_size"], random_state=cfg["split"]["random_state"], stratify=df[target]
    )
    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Train: {train.shape}  buyers {train[target].mean():.1%}")
    print(f"Test:  {test.shape}  buyers {test[target].mean():.1%}")
    print(f"Saved to {out_dir}")

    # 5. Feature schema for training and the app.
    path = save_json(build_schema(train, cfg, features), f"{cfg['paths']['artifacts_dir']}/schema.json")
    print(f"Schema saved to {path}")


if __name__ == "__main__":
    main()
