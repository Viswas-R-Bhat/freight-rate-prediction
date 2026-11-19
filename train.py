"""Train the freight rate model, evaluate it on a time-based holdout and write predictions.

Usage:
    python train.py              # evaluate + write predictions
    python train.py --eval-only  # just print holdout metrics
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DATA = Path("data")
TRAIN_FILE = DATA / "train_test.csv"
VALID_FILE = DATA / "validation.csv"
DECEMBER_FILE = DATA / "december-chart-inputs.csv"
RESULTS = Path("results")

HOLDOUT_START = "2025-09-01"   # Sep-Oct are held out, like Nov-Dec come after the training data
# Backtest error flattens after ~1,000 trees and doesn't rise up to 3,000 in any window, so a fixed count on
# that plateau replaces early stopping on a single month, which picked anywhere from 228 to 1,427 trees.
N_TREES = 1500
RATIO_MIN, RATIO_MAX = 0.45, 2.5  # $/mile vs equipment median outside this range = bad label
SEED = 42

PARAMS = dict(
    objective="huber",
    alpha=0.05,  # target is log($/mile), so this is roughly "errors above 5% are treated as outliers"
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
)


# ---------------------------------------------------------------- data prep

def daily_means(*frames):
    """Per-day average of market_index and quote_signal over all loads (features only, no labels)."""
    df = pd.concat(frames)
    return df.groupby(pd.to_datetime(df["date"]))[["market_index", "quote_signal"]].mean()


def clean(df, daily):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # negative weights have the same magnitudes as normal ones, so they're sign errors
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    df["market_index"] = df["market_index"].fillna(df["date"].map(daily["market_index"]))
    # quote_signal is within ~1% of the true $/mile in Jan-Mar, Jun and Sep but noisy and biased low in the
    # other months, Nov-Dec included. Averaged over a day's 150-200 loads it still tracks that day's market level.
    df["quote_daily"] = df["date"].map(daily["quote_signal"])
    return df


def drop_bad_labels(df):
    rpm = df["posted_rate"] / df["distance"]
    ratio = rpm / rpm.groupby(df["equipment"]).transform("median")
    return df[ratio.between(RATIO_MIN, RATIO_MAX)]


def features(df):
    # quote_signal itself is left out: a model that trusts the per-load quote breaks when training is
    # mostly clean-quote months and the forecast months are noisy (May-Jun backtest). Only its daily average is used.
    X = df[["distance", "weight", "market_index",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].copy()
    X["log_distance"] = np.log(df["distance"])
    X["equipment"] = df["equipment"].astype(pd.CategoricalDtype(["Dry Van", "Flatbed", "Reefer"]))
    X["dayofweek"] = df["date"].dt.dayofweek
    # the date's market level comes from quote_daily and market_index; dayofyear is left out because
    # trees can't extrapolate it past October (it lost on 2 of 3 backtest windows)
    X["quote_daily"] = df["quote_daily"]
    return X


def log_rpm(df):
    return np.log(df["posted_rate"] / df["distance"])


# ---------------------------------------------------------------- model

def trend_slope(train):
    """Daily drift in log($/mile) that the load features don't explain.

    Trees can't extrapolate past the last training date, so the drift is removed from the
    target before fitting and added back when predicting. Residuals are out-of-fold so the
    model can't soak up part of the trend, and quote_daily, which pins down the date, is left
    out for the same reason.
    """
    y = log_rpm(train)
    X = features(train).drop(columns="quote_daily")
    oof = np.zeros(len(train))
    for fit_idx, pred_idx in KFold(5, shuffle=True, random_state=0).split(X):
        m = lgb.LGBMRegressor(n_estimators=400, **PARAMS)
        m.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[pred_idx] = m.predict(X.iloc[pred_idx])
    return float(np.polyfit(train["date"].dt.dayofyear, y - oof, 1)[0])


def fit(train, n_trees):
    slope = trend_slope(train)
    y = log_rpm(train) - slope * train["date"].dt.dayofyear
    model = lgb.LGBMRegressor(n_estimators=n_trees, **PARAMS).fit(features(train), y)
    return model, slope


def predict(fitted, df):
    model, slope = fitted
    pred = model.predict(features(df)) + slope * df["date"].dt.dayofyear.to_numpy()
    return np.exp(pred) * df["distance"].to_numpy()


def baseline(train, test):
    """Median $/mile per equipment type times distance."""
    rpm = (train["posted_rate"] / train["distance"]).groupby(train["equipment"]).median()
    return test["equipment"].map(rpm).to_numpy() * test["distance"].to_numpy()


def scores(y, p):
    pct = np.abs(y - p) / y
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "MAPE_%": float(pct.mean() * 100),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "within_5%": float((pct < 0.05).mean() * 100),
        "within_10%": float((pct < 0.10).mean() * 100),
    }


# ---------------------------------------------------------------- main

def evaluate(data):
    train = drop_bad_labels(data[data["date"] < HOLDOUT_START])
    holdout = data[data["date"] >= HOLDOUT_START]
    is_clean = holdout.index.isin(drop_bad_labels(holdout).index)
    y = holdout["posted_rate"].to_numpy()

    preds = {"baseline": baseline(train, holdout), "lightgbm": predict(fit(train, N_TREES), holdout)}

    results = {
        "rows": {"train": int((data["date"] < HOLDOUT_START).sum()),
                 "train_after_dropping_bad_labels": len(train),
                 "holdout": len(holdout)},
        "n_trees": N_TREES,
    }
    for name, p in preds.items():
        results[name] = {"all_rows": scores(y, p), "clean_rows": scores(y[is_clean], p[is_clean])}
    return results


def predict_december(fitted, data, daily):
    dec = pd.read_csv(DECEMBER_FILE)
    X = dec.assign(date=pd.to_datetime(dec["date"]))
    lane = data[(data["pickup"] == "Lexington") & (data["delivery"] == "Fort Wayne")].iloc[0]
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[col] = lane[col]
    # not in the chart file: use December's observed daily market and quote levels
    X["market_index"] = X["date"].map(daily["market_index"])
    X["quote_daily"] = X["date"].map(daily["quote_signal"])
    dec["predicted_rate"] = predict(fitted, X).round(2)
    dec.to_csv(DECEMBER_FILE, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    raw, valid_raw = pd.read_csv(TRAIN_FILE), pd.read_csv(VALID_FILE)
    daily = daily_means(raw, valid_raw)
    data = clean(raw, daily)

    results = evaluate(data)
    print(json.dumps(results, indent=2))
    if args.eval_only:
        return

    fitted = fit(drop_bad_labels(data), N_TREES)
    results["trend_pct_per_30_days"] = fitted[1] * 30 * 100

    valid = clean(valid_raw, daily)
    out = pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": predict(fitted, valid).round(2)})
    out.to_csv("validation_predictions.csv", index=False)
    predict_december(fitted, data, daily)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print("saved validation_predictions.csv, data/december-chart-inputs.csv, results/metrics.json")


if __name__ == "__main__":
    main()
