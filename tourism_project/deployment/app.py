"""Streamlit app: will this customer buy the Wellness Tourism Package?

Run locally:  streamlit run tourism_project/deployment/app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
# In the GitHub repo the model sits in tourism_project/model; in a Hugging Face bundle it sits next to app.py.
BASE = HERE.parent if (HERE.parent / "model" / "model.joblib").exists() else HERE
MODEL_PATH = BASE / "model" / "model.joblib"
SCHEMA_PATH = BASE / "artifacts" / "schema.json"
METRICS_PATH = BASE / "artifacts" / "metrics.json"

DESCRIPTIONS = {
    "Age": "Age of the customer",
    "TypeofContact": "How the customer was contacted",
    "CityTier": "City development tier (1 = most developed)",
    "Occupation": "Customer's occupation",
    "Gender": "Gender of the customer",
    "NumberOfPersonVisiting": "Total people travelling with the customer",
    "PreferredPropertyStar": "Preferred hotel star rating",
    "MaritalStatus": "Marital status",
    "NumberOfTrips": "Average trips per year",
    "Passport": "Holds a valid passport",
    "OwnCar": "Owns a car",
    "NumberOfChildrenVisiting": "Children under 5 travelling",
    "Designation": "Job designation",
    "MonthlyIncome": "Gross monthly income",
    "PitchSatisfactionScore": "Pitch satisfaction score (1-5)",
    "ProductPitched": "Product pitched to the customer",
    "NumberOfFollowups": "Follow-ups after the pitch",
    "DurationOfPitch": "Duration of the sales pitch (minutes)",
}

st.set_page_config(page_title="Wellness Package Predictor", page_icon=":airplane:", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def clean_frame(df, schema):
    """Same text clean-up as the training pipeline (trim + spelling fixes) for uploaded CSVs."""
    df = df.copy()
    for col, spec in schema["columns"].items():
        if col in df.columns and spec["type"] == "categorical":
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].replace(schema["category_map"].get(col, {}))
    return df


def input_widget(col, spec):
    help_text = DESCRIPTIONS.get(col)
    if spec["type"] == "categorical":
        return st.selectbox(col, spec["values"], index=spec["values"].index(spec["default"]), help=help_text, key=col)
    if spec["binary"]:
        return st.selectbox(col, [0, 1], index=spec["default"], format_func=lambda v: "Yes" if v else "No",
                            help=help_text, key=col)
    if spec["integer"] and spec["max"] - spec["min"] <= 20:
        return st.slider(col, spec["min"], spec["max"], spec["default"], step=1, help=help_text, key=col)
    if spec["integer"]:
        return st.number_input(col, min_value=spec["min"], max_value=spec["max"], value=spec["default"],
                               step=1, help=help_text, key=col)
    return st.number_input(col, min_value=float(spec["min"]), max_value=float(spec["max"]),
                           value=float(spec["default"]), help=help_text, key=col)


if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
    st.error("No trained model found. Run the GitHub Actions pipeline (or train.py) first.")
    st.stop()

model = load_model()
schema = load_json(SCHEMA_PATH)
metrics = load_json(METRICS_PATH) if METRICS_PATH.exists() else {}
features = list(model.feature_names_in_)
threshold = metrics.get("threshold", 0.5)

with st.sidebar:
    st.header("Model")
    if metrics:
        best = metrics["selected_model"]
        test = metrics["models"][best]["test"]
        st.write(f"**{best}**, trained on {metrics['feature_set']}")
        st.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
        st.metric("Test recall", f"{test['recall']:.3f}")
        st.caption(f"'Will Buy' when the purchase probability is at least {threshold:.0%}.")

st.title("Wellness Tourism Package Predictor")
st.write("Visit with Us: find the customers most likely to buy the Wellness Tourism Package before contacting them.")

single_tab, batch_tab = st.tabs(["Single customer", "Batch scoring (CSV)"])

with single_tab:
    with st.form("customer"):
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(features):
            with cols[i % 3]:
                values[col] = input_widget(col, schema["columns"][col])
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        input_df = pd.DataFrame([values], columns=features)  # inputs saved into a DataFrame
        st.subheader("Customer details")
        st.dataframe(input_df, hide_index=True)
        proba = float(model.predict_proba(input_df)[0, 1])
        if proba >= threshold:
            st.success(f"**Will Buy** - purchase probability {proba:.1%}")
        else:
            st.info(f"**Will Not Buy** - purchase probability {proba:.1%}")
        st.progress(proba)

with batch_tab:
    st.write(f"Upload a CSV with these columns: {', '.join(features)}. Extra columns such as CustomerID are kept.")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        raw = pd.read_csv(upload)
        missing = [c for c in features if c not in raw.columns]
        if missing:
            st.error(f"The file is missing these columns: {', '.join(missing)}")
        else:
            proba = model.predict_proba(clean_frame(raw, schema)[features])[:, 1]
            out = raw.assign(PurchaseProbability=proba.round(4),
                             Prediction=["Will Buy" if p >= threshold else "Will Not Buy" for p in proba])
            st.write(f"Scored {len(out)} customers; {int((proba >= threshold).sum())} predicted to buy.")
            st.dataframe(out, hide_index=True)
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
