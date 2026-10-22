"""Step 3 - Tune, compare and track models with MLflow; save the best pipeline.

Usage: python tourism_project/model_building/train.py
"""
import os
import time
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import load_config, load_json, parse_args, resolve, save_json


def candidate_models(seed, pos_weight):
    """Model name -> (estimator, hyper-parameter search space, uses sample weights)."""
    return {
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            {"clf__max_depth": [3, 5, 7, 10, 15, None], "clf__min_samples_split": [2, 5, 10, 20],
             "clf__min_samples_leaf": [1, 2, 5, 10], "clf__criterion": ["gini", "entropy"]},
            False,
        ),
        "Bagging": (
            BaggingClassifier(DecisionTreeClassifier(class_weight="balanced", random_state=seed), random_state=seed),
            {"clf__n_estimators": [50, 100, 200], "clf__max_samples": [0.6, 0.8, 1.0],
             "clf__max_features": [0.6, 0.8, 1.0], "clf__bootstrap": [True, False]},
            False,
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed),
            {"clf__n_estimators": [100, 200, 300, 500], "clf__max_depth": [5, 10, 15, 20, None],
             "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", "log2", 0.5]},
            False,
        ),
        "AdaBoost": (
            AdaBoostClassifier(algorithm="SAMME", random_state=seed),
            {"clf__n_estimators": [50, 100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]},
            True,
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=seed),
            {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
             "clf__max_depth": [2, 3, 4, 5], "clf__subsample": [0.8, 1.0]},
            True,
        ),
        "XGBoost": (
            XGBClassifier(random_state=seed, eval_metric="logloss", scale_pos_weight=pos_weight, n_jobs=1),
            {"clf__n_estimators": [100, 200, 300, 500], "clf__max_depth": [3, 4, 5, 6, 8],
             "clf__learning_rate": [0.01, 0.05, 0.1, 0.2], "clf__subsample": [0.7, 0.8, 1.0],
             "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5]},
            False,
        ),
    }


def build_pipeline(estimator, num_cols, cat_cols):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    return Pipeline([("pre", pre), ("clf", estimator)])


def evaluate(model, X, y, threshold):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def split_columns(schema, exclude_post_contact):
    drop = set(schema["post_contact_columns"]) if exclude_post_contact else set()
    features = [c for c in schema["features"] if c not in drop]
    num = [c for c in features if schema["columns"][c]["type"] == "numeric"]
    cat = [c for c in features if schema["columns"][c]["type"] == "categorical"]
    return features, num, cat


def setup_mlflow(cfg):
    """Track experiments in a local SQLite store inside tourism_project/mlflow/."""
    tcfg = cfg["tracking"]
    db = resolve(tcfg["db_path"])
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + db.as_posix())
    if mlflow.get_experiment_by_name(tcfg["experiment"]) is None:
        art = resolve(tcfg["artifact_dir"])
        art.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(tcfg["experiment"], artifact_location=art.as_uri())
    mlflow.set_experiment(tcfg["experiment"])


def short(params):
    return {k.replace("clf__", ""): v for k, v in params.items()}


def write_ci_summary(results, best_name, leakage):
    """Show the comparison table on the GitHub Actions run page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["## Model comparison", "", "| Model | CV ROC-AUC | Test ROC-AUC | Test F1 | Test Recall |",
             "|---|---|---|---|---|"]
    for name, r in results.items():
        mark = " **(selected)**" if name == best_name else ""
        lines.append(f"| {name}{mark} | {r['cv_score']} | {r['test']['roc_auc']} | {r['test']['f1']} | {r['test']['recall']} |")
    lines += ["", f"Leakage check ({leakage['feature_set']}): test ROC-AUC {leakage['test']['roc_auc']}"]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    args = parse_args("Train and tune Tourism package models")
    cfg = load_config(args)
    tcfg, target = cfg["training"], cfg["target"]
    seed = cfg["split"]["random_state"]
    artifacts = cfg["paths"]["artifacts_dir"]

    # Load the train/test split produced by the preparation step (workflow artifact in CI).
    processed = resolve(cfg["data"]["processed_dir"])
    train = pd.read_csv(processed / "train.csv")
    test = pd.read_csv(processed / "test.csv")
    schema = load_json(f"{artifacts}/schema.json")

    exclude = cfg["features"]["exclude_post_contact"]
    feature_set = "pre-contact only" if exclude else "all features"
    features, num_cols, cat_cols = split_columns(schema, exclude)
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    print(f"Feature set: {feature_set} ({len(features)} columns): {features}\n")

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    sample_weight = compute_sample_weight("balanced", y_train)
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=seed)

    setup_mlflow(cfg)
    results, fitted = {}, {}
    with mlflow.start_run(run_name=f"training-{datetime.now():%Y%m%d-%H%M%S}") as parent:
        mlflow.log_params({"feature_set": feature_set, "n_features": len(features), "cv_folds": tcfg["cv_folds"],
                           "n_iter": tcfg["n_iter"], "scoring": tcfg["scoring"], "threshold": tcfg["threshold"],
                           "train_rows": len(train), "test_rows": len(test)})

        for name, (estimator, space, use_weights) in candidate_models(seed, pos_weight).items():
            start = time.time()
            with mlflow.start_run(run_name=name, nested=True):
                search = RandomizedSearchCV(
                    build_pipeline(estimator, num_cols, cat_cols), space, n_iter=tcfg["n_iter"],
                    scoring=tcfg["scoring"], cv=cv, n_jobs=tcfg["n_jobs"], random_state=seed, refit=True,
                )
                search.fit(X_train, y_train, **({"clf__sample_weight": sample_weight} if use_weights else {}))

                # Log every tuned parameter combination as its own nested run.
                cvr = pd.DataFrame(search.cv_results_)
                for i, params in enumerate(cvr["params"]):
                    with mlflow.start_run(run_name=f"{name}-candidate-{i + 1:02d}", nested=True):
                        mlflow.log_params(short(params))
                        mlflow.log_metrics({"cv_mean": cvr.loc[i, "mean_test_score"],
                                            "cv_std": cvr.loc[i, "std_test_score"]})

                best = search.best_estimator_
                result = {
                    "best_params": short(search.best_params_),
                    "cv_score": round(search.best_score_, 4),
                    "train": evaluate(best, X_train, y_train, tcfg["threshold"]),
                    "test": evaluate(best, X_test, y_test, tcfg["threshold"]),
                    "fit_seconds": round(time.time() - start, 1),
                }
                mlflow.log_params(result["best_params"])
                mlflow.log_metric("cv_best", result["cv_score"])
                mlflow.log_metrics({f"train_{k}": v for k, v in result["train"].items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in result["test"].items()})
                cv_path = resolve(f"{artifacts}/cv_results_{name}.csv")
                cvr.drop(columns=["params"]).to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path))

            results[name], fitted[name] = result, (best, search.best_params_)
            print(f"{name:<17} CV ROC-AUC={result['cv_score']:.4f} | test ROC-AUC={result['test']['roc_auc']:.4f} "
                  f"F1={result['test']['f1']:.4f} recall={result['test']['recall']:.4f} ({result['fit_seconds']}s)")
            print(f"{'':<17} best params: {result['best_params']}")

        # Select on cross-validated score only; the test set is used purely for reporting.
        best_name = max(results, key=lambda n: results[n]["cv_score"])
        best_model, best_params = fitted[best_name]
        print(f"\nSelected model: {best_name}")

        # Leakage check: refit the winner on the other feature set and compare.
        alt_features, alt_num, alt_cat = split_columns(schema, not exclude)
        estimator, _, use_weights = candidate_models(seed, pos_weight)[best_name]
        alt = build_pipeline(estimator, alt_num, alt_cat).set_params(**best_params)
        alt.fit(train[alt_features], y_train, **({"clf__sample_weight": sample_weight} if use_weights else {}))
        leakage = {"model": best_name, "feature_set": "all features" if exclude else "pre-contact only",
                   "test": evaluate(alt, test[alt_features], y_test, tcfg["threshold"])}
        print(f"Leakage check - {best_name} with {leakage['feature_set']}: test ROC-AUC={leakage['test']['roc_auc']:.4f}")

        model_path = resolve(cfg["paths"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)

        metrics = {"selected_model": best_name, "selection_metric": f"cv_{tcfg['scoring']}",
                   "feature_set": feature_set, "features": features, "threshold": tcfg["threshold"],
                   "train_rows": len(train), "test_rows": len(test), "mlflow_run_id": parent.info.run_id,
                   "models": results, "leakage_check": leakage}
        metrics_path = save_json(metrics, f"{artifacts}/metrics.json")
        comparison = pd.DataFrame([
            {"model": n, "cv_roc_auc": r["cv_score"], **{f"test_{k}": v for k, v in r["test"].items()},
             **{f"train_{k}": v for k, v in r["train"].items()}}
            for n, r in results.items()
        ]).sort_values("cv_roc_auc", ascending=False)
        comparison.to_csv(resolve(f"{artifacts}/model_comparison.csv"), index=False)

        mlflow.set_tag("selected_model", best_name)
        mlflow.log_metrics({f"best_test_{k}": v for k, v in results[best_name]["test"].items()})
        mlflow.log_metric("leakage_check_test_roc_auc", leakage["test"]["roc_auc"])
        for path in (model_path, metrics_path):
            mlflow.log_artifact(str(path))

    write_ci_summary(results, best_name, leakage)
    print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    main()
