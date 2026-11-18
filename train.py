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
EARLY_STOP_START = "2025-08-01"  # August is used to pick the number of trees
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

def daily_market_index(*frames):
    df = pd.concat(frames)
    return df.groupby(pd.to_datetime(df["date"]))["market_index"].mean()


def clean(df, daily_mi):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # negative weights have the same magnitudes as normal ones, so they're sign errors
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))
    df["market_index"] = df["market_index"].fillna(df["date"].map(daily_mi))
    return df


def drop_bad_labels(df):
    rpm = df["posted_rate"] / df["distance"]
    ratio = rpm / rpm.groupby(df["equipment"]).transform("median")
    return df[ratio.between(RATIO_MIN, RATIO_MAX)]


def features(df):
    X = df[["distance", "weight", "market_index", "quote_signal",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].copy()
    X["log_distance"] = np.log(df["distance"])
    X["equipment"] = df["equipment"].astype(pd.CategoricalDtype(["Dry Van", "Flatbed", "Reefer"]))
    X["dayofweek"] = df["date"].dt.dayofweek
    X["dayofyear"] = df["date"].dt.dayofyear
    return X


def log_rpm(df):
    return np.log(df["posted_rate"] / df["distance"])


# ---------------------------------------------------------------- model

def trend_slope(train):
    """Daily drift in log($/mile) that the load features don't explain.

    Trees can't extrapolate past the last training date, so the drift is removed from the
    target before fitting and added back when predicting. Residuals are out-of-fold so the
    model can't soak up part of the trend.
    """
    y = log_rpm(train)
    X = features(train).drop(columns="dayofyear")
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


def pick_n_trees(train):
    early, late = train[train["date"] < EARLY_STOP_START], train[train["date"] >= EARLY_STOP_START]
    m = lgb.LGBMRegressor(n_estimators=5000, **PARAMS)
    m.fit(features(early), log_rpm(early),
          eval_set=[(features(late), log_rpm(late))],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.best_iteration_


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

    n_trees = pick_n_trees(train)
    preds = {"baseline": baseline(train, holdout), "lightgbm": predict(fit(train, n_trees), holdout)}

    results = {
        "rows": {"train": int((data["date"] < HOLDOUT_START).sum()),
                 "train_after_dropping_bad_labels": len(train),
                 "holdout": len(holdout)},
        "n_trees": n_trees,
    }
    for name, p in preds.items():
        results[name] = {"all_rows": scores(y, p), "clean_rows": scores(y[is_clean], p[is_clean])}
    return results


def predict_december(fitted, data, daily_mi):
    dec = pd.read_csv(DECEMBER_FILE)
    X = dec.assign(date=pd.to_datetime(dec["date"]))
    lane = data[(data["pickup"] == "Lexington") & (data["delivery"] == "Fort Wayne")].iloc[0]
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[col] = lane[col]
    # these two aren't in the chart file: use December's observed market level and a typical quote
    X["market_index"] = X["date"].map(daily_mi)
    X["quote_signal"] = data["quote_signal"].median()
    dec["predicted_rate"] = predict(fitted, X).round(2)
    dec.to_csv(DECEMBER_FILE, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    raw, valid_raw = pd.read_csv(TRAIN_FILE), pd.read_csv(VALID_FILE)
    daily_mi = daily_market_index(raw, valid_raw)
    data = clean(raw, daily_mi)

    results = evaluate(data)
    print(json.dumps(results, indent=2))
    if args.eval_only:
        return

    fitted = fit(drop_bad_labels(data), results["n_trees"])
    results["trend_pct_per_30_days"] = fitted[1] * 30 * 100

    valid = clean(valid_raw, daily_mi)
    out = pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": predict(fitted, valid).round(2)})
    out.to_csv("validation_predictions.csv", index=False)
    predict_december(fitted, data, daily_mi)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print("saved validation_predictions.csv, data/december-chart-inputs.csv, results/metrics.json")


if __name__ == "__main__":
    main()
