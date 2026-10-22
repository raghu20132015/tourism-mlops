# Tourism Package Prediction - Visit with Us

MLOps pipeline that predicts whether a customer will buy the Wellness Tourism Package before they are contacted.

| Folder | Contents |
|---|---|
| `tourism_project/data/` | `Tourism.csv` (input dataset) |
| `tourism_project/model_building/` | `data_registration.py`, `data_preparation.py`, `train.py`, `common.py` |
| `tourism_project/model/` | `model.joblib`, committed by the pipeline |
| `tourism_project/artifacts/` | schema, metrics and model comparison |
| `tourism_project/deployment/` | Streamlit `app.py`, `requirements.txt`, `Dockerfile` |
| `.github/workflows/pipeline.yml` | GitHub Actions pipeline |

Run locally:

    pip install -r requirements.txt
    python tourism_project/model_building/data_registration.py
    python tourism_project/model_building/data_preparation.py
    python tourism_project/model_building/train.py
    streamlit run tourism_project/deployment/app.py
