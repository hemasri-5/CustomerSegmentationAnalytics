# Customer Segmentation and Predictive Analytics System

A beginner-friendly team project using Python, Streamlit, Pandas, Scikit-learn, and Matplotlib/Seaborn.

## Features
- Loads a CSV or Excel retail transaction dataset.
- Cleans missing/invalid records and calculates sales.
- Creates customer-level RFM (Recency, Frequency, Monetary) features.
- Groups customers with K-Means clustering.
- Forecasts weekly sales with a Random Forest model using lag features.
- Evaluates forecasts on a time-ordered holdout period (MAE, RMSE, R²).
- Provides an interactive Streamlit dashboard, charts, downloads, and business recommendations.

## Quick start (Windows + VS Code)

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. The app opens in your browser. It loads `sample_transactions.csv` automatically.
5. To use another dataset, upload a CSV/XLSX file in the sidebar.

## Dataset format

Required columns (names must match):
- `InvoiceNo`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`

The UCI Online Retail dataset uses these columns. Download it from:
https://archive.ics.uci.edu/dataset/352/online+retail

The included `sample_transactions.csv` is synthetic demo data generated for this project. It is not real customer information.

## Team of five — suggested ownership

1. **Team Leader:** integrate modules, own `app.py`, coordinate testing, prepare demo/report.
2. **Data preprocessing:** own `preprocessing.py`, validate columns, cleaning, document data quality.
3. **Customer segmentation:** own `segmentation.py`, RFM logic, K-Means, segment interpretation.
4. **Predictive analytics:** own `prediction.py`, time split, sales forecast, evaluation.
5. **Visualization and insights:** own `visualizations.py`, chart quality, business recommendations and screenshots.

Everyone should review and understand the final integrated system. The leader should agree on function names and output formats before integration.

## Files
- `app.py` — main dashboard.
- `preprocessing.py` — input and data cleaning.
- `segmentation.py` — RFM metrics and K-Means.
- `prediction.py` — weekly sales forecasting and evaluation.
- `visualizations.py` — chart functions.
- `sample_transactions.csv` — synthetic demo transactions.
- `requirements.txt` — dependencies.

## Important limitations
- The demo data is synthetic and only demonstrates the workflow.
- K-Means segment IDs are arbitrary; descriptive labels are heuristic and should be reviewed against each dataset.
- The sales model is a baseline. Compare it with a simple baseline forecast and use more history/features for a serious evaluation.
- Forecasting is not churn prediction. Churn requires a defensible churn definition and suitable labeled data.
- Remove or anonymize personal data before sharing a dataset with your team.
