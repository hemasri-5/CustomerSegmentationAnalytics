"""Weekly sales forecasting using a time-ordered train/test split."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def weekly_sales(df):
    sales = df.set_index("InvoiceDate")["Sales"].resample("W").sum().rename("Sales").reset_index()
    if len(sales) == 0:
        return sales
    # Fill missing weeks with zero sales so lags remain aligned.
    full_index = pd.date_range(sales["InvoiceDate"].min(), sales["InvoiceDate"].max(), freq="W")
    sales = sales.set_index("InvoiceDate").reindex(full_index, fill_value=0).rename_axis("InvoiceDate").reset_index()
    return sales

def _features(sales):
    data = sales.copy().sort_values("InvoiceDate").reset_index(drop=True)
    data["time_index"] = np.arange(len(data))
    for lag in [1, 2, 3, 4]:
        data[f"lag_{lag}"] = data["Sales"].shift(lag)
    data["rolling_mean_4"] = data["Sales"].shift(1).rolling(4).mean()
    data["week_of_year"] = data["InvoiceDate"].dt.isocalendar().week.astype(int)
    data["month"] = data["InvoiceDate"].dt.month
    return data.dropna().reset_index(drop=True)

def forecast_sales(df, forecast_periods=4):
    """Train on earlier weeks, evaluate on the last 20% and forecast future weeks."""
    sales = weekly_sales(df)
    if len(sales) < 12:
        raise ValueError("At least 12 weeks of transaction history are required for sales forecasting.")
    data = _features(sales)
    if len(data) < 8:
        raise ValueError("Not enough usable weekly data after creating lag features.")
    feature_cols = ["time_index", "lag_1", "lag_2", "lag_3", "lag_4",
                    "rolling_mean_4", "week_of_year", "month"]
    split = max(1, int(len(data) * 0.8))
    if split >= len(data):
        split = len(data) - 1
    train, test = data.iloc[:split], data.iloc[split:]
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    model.fit(train[feature_cols], train["Sales"])
    test_pred = np.maximum(0, model.predict(test[feature_cols]))
    mae = mean_absolute_error(test["Sales"], test_pred)
    rmse = float(np.sqrt(mean_squared_error(test["Sales"], test_pred)))
    r2 = r2_score(test["Sales"], test_pred) if len(test) > 1 else float("nan")
    evaluation = {"MAE": float(mae), "RMSE": rmse, "R2": float(r2), "Test weeks": int(len(test))}

    # Refit on all usable history for recursive multi-step forecasts.
    final_model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
    final_model.fit(data[feature_cols], data["Sales"])
    history = sales.copy()
    forecasts = []
    for _ in range(max(1, int(forecast_periods))):
        next_date = history["InvoiceDate"].max() + pd.Timedelta(days=7)
        vals = history["Sales"].tolist()
        lag1, lag2, lag3, lag4 = vals[-1], vals[-2], vals[-3], vals[-4]
        roll4 = float(np.mean(vals[-4:]))
        feat = pd.DataFrame([{
            "time_index": len(history),
            "lag_1": lag1, "lag_2": lag2, "lag_3": lag3, "lag_4": lag4,
            "rolling_mean_4": roll4,
            "week_of_year": int(next_date.isocalendar().week),
            "month": next_date.month
        }])
        pred = max(0.0, float(final_model.predict(feat[feature_cols])[0]))
        forecasts.append({"InvoiceDate": next_date, "PredictedSales": pred})
        history = pd.concat([history, pd.DataFrame([{"InvoiceDate": next_date, "Sales": pred}])], ignore_index=True)
    return sales, test[["InvoiceDate", "Sales"]].assign(PredictedSales=test_pred), pd.DataFrame(forecasts), evaluation
