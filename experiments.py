"""Ablations on the Sep-Oct holdout. Each row changes one thing relative to the final model.

    python experiments.py   # writes results/experiments.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

import train as T


def haversine_miles(df):
    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in
                              ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def lane_features(df):
    X = T.features(df)
    X["straight_line_miles"] = haversine_miles(df)
    X["circuity"] = df["distance"] / X["straight_line_miles"].clip(lower=1)
    return X


def run(train, test, n_trees, feats=T.features, params=None, trend=True, weight=None, model="lgbm"):
    slope = T.trend_slope(train) if trend else 0.0
    t_train = train["date"].dt.dayofyear.to_numpy()
    y = T.log_rpm(train) - slope * t_train
    if model == "lgbm":
        m = lgb.LGBMRegressor(n_estimators=n_trees, **{**T.PARAMS, **(params or {})})
        m.fit(feats(train), y, sample_weight=weight)
    else:
        m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05,
                                          categorical_features="from_dtype", random_state=0)
        m.fit(feats(train), y)
    pred = m.predict(feats(test)) + slope * test["date"].dt.dayofyear.to_numpy()
    return np.exp(pred) * test["distance"].to_numpy()


def main():
    raw, valid_raw = pd.read_csv(T.TRAIN_FILE), pd.read_csv(T.VALID_FILE)
    data = T.clean(raw, T.daily_market_index(raw, valid_raw))
    train = T.drop_bad_labels(data[data["date"] < T.HOLDOUT_START])
    holdout = data[data["date"] >= T.HOLDOUT_START]
    is_clean = holdout.index.isin(T.drop_bad_labels(holdout).index)
    y = holdout["posted_rate"].to_numpy()
    n = T.pick_n_trees(train)

    age = (train["date"].max() - train["date"]).dt.days.to_numpy()
    no_doy = lambda df: T.features(df).drop(columns="dayofyear")

    variants = {
        "final model": lambda: run(train, holdout, n),
        "no trend term": lambda: run(train, holdout, n, trend=False),
        "L1 loss instead of Huber": lambda: run(train, holdout, n, params={"objective": "l1"}),
        "no day-of-year feature": lambda: run(train, holdout, n, feats=no_doy),
        "recency weights (60-day half-life)": lambda: run(train, holdout, n, weight=0.5 ** (age / 60)),
        "+ straight-line distance, circuity": lambda: run(train, holdout, n, feats=lane_features),
        "bigger trees (63 leaves, 1200 trees)": lambda: run(
            train, holdout, 1200, params={"num_leaves": 63, "learning_rate": 0.02}),
        "sklearn HistGradientBoosting": lambda: run(train, holdout, n, model="hgb"),
    }

    rows = {}
    for name, make in variants.items():
        p = make()
        pct = np.abs(y - p) / y
        rows[name] = {
            "MAE": np.mean(np.abs(y - p)),
            "MAPE_%": pct.mean() * 100,
            "within_5%": (pct < 0.05).mean() * 100,
            "MAE_clean": np.mean(np.abs(y - p)[is_clean]),
            "bias_%": (np.median(p[is_clean] / y[is_clean]) - 1) * 100,
        }
        print(f"{name:40s} MAE {rows[name]['MAE']:7.2f}")

    table = pd.DataFrame(rows).T.round(2)
    T.RESULTS.mkdir(exist_ok=True)
    table.to_csv(T.RESULTS / "experiments.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
