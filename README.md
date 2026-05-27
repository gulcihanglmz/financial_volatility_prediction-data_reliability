Financial Volatility Prediction: Data Reliability Scoring and Chronological Validation
======================================

Purpose
-------
VolPred is a compact Streamlit dashboard for forecasting financial asset volatility, comparing model performance, and monitoring input data quality. It is intended as a prototype monitoring and decision‑support tool, not a production trading engine.


https://github.com/user-attachments/assets/40aa833b-55b8-4540-b202-d505d180623b


Features
--------
- 30‑day volatility forecasts with 80% and 95% confidence intervals.
- Model comparison table and leaderboard (Accuracy, Precision, Recall, F1).
- Asset‑specific data reliability score computed with IsolationForest.
- Mock data fallback when live data or models are unavailable.

Requirements
------------
- Python 3.9+ recommended.
- See `fintech_dashboard/requirements.txt` for required packages (Streamlit, pandas, numpy, plotly, scikit‑learn, yfinance, joblib, etc.).

Quick start
-----------
1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r fintech_dashboard/requirements.txt
```

2. Run the dashboard:

```bash
streamlit run fintech_dashboard/app.py --server.port 8501
```

How the app decides data and models
----------------------------------
- If `models/volatility_best_model.pkl` and `models/volatility_scaler.pkl` exist and `joblib` is available, the app will attempt to load and use them for live model outputs.
- If model files or `joblib` are missing, the app falls back to deterministic mock data produced by `fintech_dashboard/data/mock_data.py`.
- If `data/volatility_model_comparison.csv` exists, `mock_data.py` will read model metrics from that file and the dashboard will display all rows found there. Otherwise a built‑in fallback list is used.

Project layout (important files)
--------------------------------
- `fintech_dashboard/app.py` — main Streamlit app and UI logic.
- `fintech_dashboard/data/mock_data.py` — mock data and metric generation; reads `data/volatility_model_comparison.csv` if present.
- `data/volatility_model_comparison.csv` — optional CSV with model metrics (Accuracy, Precision, Recall, F1). Place one row per model.
- `models/` — optional: saved model and scaler files for live predictions.
- `utils/` — plotting utilities and export helpers.
- `src/` — feature engineering and data processing code.

Common tasks
------------
- Show seven models in the dashboard: put seven rows in `data/volatility_model_comparison.csv` (columns: Model, Accuracy, Precision, Recall, F1). The app will load and display them.
- Force mock mode: remove or rename files in `models/` so `load_trained_model()` cannot find them.

Troubleshooting
---------------
- Model load errors: check `models/` and that `joblib` is installed. `load_trained_model()` returns a status message shown in the sidebar.
- API data errors: if `yfinance` returns too short or malformed data, the app will fall back to mock data and log the cause in the sidebar status.
- If the dashboard shows identical mock data across assets, restart the app — asset seeding uses a hash of the asset label; a restart ensures consistent mock variety.

Notes for thesis / presentation
-----------------------------
- The data reliability score is stored in `vol_df["data_reliability"]` and visualized on the Data Reliability tab.
- Forecast confidence intervals are produced by `generate_future_forecast()` in mock mode; in production replace with your model ensemble or bootstrap procedure.

Contributing and contact
------------------------
This repository contains private work. For questions or collaboration, contact the project owner.

License
-------
Proprietary — contact the owner for reuse or contributions.
