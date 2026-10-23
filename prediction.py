"""Weekly sales forecasting with chronological evaluation and a baseline."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLS = [
    "time_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "rolling_mean_4",
    "week_of_year",
    "month",
]


def weekly_sales(df):
    """Aggregate transaction sales into continuous weekly totals."""
    required = {"InvoiceDate", "Sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = df[["InvoiceDate", "Sales"]].copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate", "Sales"])

    if data.empty:
        return pd.DataFrame(columns=["InvoiceDate", "Sales"])

    sales = (
        data.set_index("InvoiceDate")["Sales"]
        .resample("W")
        .sum()
        .rename("Sales")
        .reset_index()
    )

    full_index = pd.date_range(
        sales["InvoiceDate"].min(), sales["InvoiceDate"].max(), freq="W"
    )
    sales = (
        sales.set_index("InvoiceDate")
        .reindex(full_index, fill_value=0)
        .rename_axis("InvoiceDate")
        .reset_index()
    )
    return sales


def _features(sales):
    """Build training features; time_index matches the original weekly series."""
    data = sales.copy().sort_values("InvoiceDate").reset_index(drop=True)
    data["time_index"] = np.arange(len(data))

    for lag in (1, 2, 3, 4):
        data[f"lag_{lag}"] = data["Sales"].shift(lag)

    data["rolling_mean_4"] = data["Sales"].shift(1).rolling(4).mean()
    data["week_of_year"] = data["InvoiceDate"].dt.isocalendar().week.astype(int)
    data["month"] = data["InvoiceDate"].dt.month
    return data.dropna().reset_index(drop=True)


def _make_next_week_features(history):
    """Build features for the week after the latest history row."""
    history = history.sort_values("InvoiceDate").reset_index(drop=True)
    if len(history) < 4:
        raise ValueError("At least 4 weeks of history are required.")

    next_date = history["InvoiceDate"].iloc[-1] + pd.Timedelta(days=7)
    values = history["Sales"].to_numpy(dtype=float)

    # The time index is the zero-based index of the next week in the full series.
    row = {
        "time_index": len(history),
        "lag_1": values[-1],
        "lag_2": values[-2],
        "lag_3": values[-3],
        "lag_4": values[-4],
        "rolling_mean_4": float(np.mean(values[-4:])),
        "week_of_year": int(next_date.isocalendar().week),
        "month": int(next_date.month),
    }
    return next_date, pd.DataFrame([row], columns=FEATURE_COLS)


def _recursive_predictions(model, history, periods):
    """Forecast multiple weeks without feeding actual future sales into later steps."""
    history = history[["InvoiceDate", "Sales"]].copy().reset_index(drop=True)
    predictions = []

    for _ in range(periods):
        next_date, features = _make_next_week_features(history)
        predicted_sales = max(0.0, float(model.predict(features)[0]))
        predictions.append({
            "InvoiceDate": next_date,
            "PredictedSales": predicted_sales,
        })
        history = pd.concat(
            [
                history,
                pd.DataFrame([{
                    "InvoiceDate": next_date,
                    "Sales": predicted_sales,
                }]),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions, columns=["InvoiceDate", "PredictedSales"])


def _calculate_metrics(actual, predicted):
    """Return MAE, RMSE, and R-squared."""
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    return mae, rmse, r2


def _recursive_last_value_baseline(history, periods):
    """Baseline that repeats the last observed sales value for each future week."""
    history = history[["InvoiceDate", "Sales"]].copy().reset_index(drop=True)
    predictions = []

    for _ in range(periods):
        next_date = history["InvoiceDate"].iloc[-1] + pd.Timedelta(days=7)
        prediction = max(0.0, float(history["Sales"].iloc[-1]))
        predictions.append(prediction)
        history = pd.concat(
            [
                history,
                pd.DataFrame([{"InvoiceDate": next_date, "Sales": prediction}]),
            ],
            ignore_index=True,
        )
    return predictions


def forecast_sales(df, forecast_periods=4):
    """
    Evaluate recursive predictions on a final chronological holdout.

    Returns: sales, evaluation_data, forecasts, evaluation
    """
    sales = weekly_sales(df)
    if len(sales) < 12:
        raise ValueError(
            "At least 12 weeks of transaction history are required for sales forecasting."
        )

    data = _features(sales)
    if len(data) < 8:
        raise ValueError("Not enough usable weekly data after creating lag features.")

    split = max(1, int(len(data) * 0.8))
    split = min(split, len(data) - 1)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(train[FEATURE_COLS], train["Sales"])

    # Use history only through the last training target date.
    last_train_date = train["InvoiceDate"].iloc[-1]
    test_history = sales.loc[
        sales["InvoiceDate"] <= last_train_date, ["InvoiceDate", "Sales"]
    ].copy()

    test_periods = len(test)
    recursive_test = _recursive_predictions(model, test_history, test_periods)

    # Verify prediction weeks correspond exactly to held-out weeks.
    actual_dates = pd.to_datetime(test["InvoiceDate"]).reset_index(drop=True)
    predicted_dates = pd.to_datetime(recursive_test["InvoiceDate"]).reset_index(drop=True)
    if not actual_dates.equals(predicted_dates):
        raise ValueError(
            "Forecast and test dates do not align. Check the weekly date series."
        )

    actual = test["Sales"].to_numpy(dtype=float)
    predicted = recursive_test["PredictedSales"].to_numpy(dtype=float)
    mae, rmse, r2 = _calculate_metrics(actual, predicted)

    baseline_predictions = _recursive_last_value_baseline(test_history, test_periods)
    baseline_mae, baseline_rmse, _ = _calculate_metrics(actual, baseline_predictions)

    evaluation = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Test weeks": int(test_periods),
        "Baseline MAE": baseline_mae,
        "Baseline RMSE": baseline_rmse,
    }

    evaluation_data = pd.DataFrame({
        "InvoiceDate": actual_dates,
        "Sales": actual,
        "PredictedSales": predicted,
        "BaselineSales": baseline_predictions,
    })

    # Refit using all known history for the future forecast.
    final_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
    )
    final_model.fit(data[FEATURE_COLS], data["Sales"])

    periods = max(1, int(forecast_periods))
    forecasts = _recursive_predictions(
        final_model,
        sales[["InvoiceDate", "Sales"]],
        periods,
    )
    return sales, evaluation_data, forecasts, evaluation
