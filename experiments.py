"""Ablations: each row changes one thing relative to the final model.

Every variant is backtested on three rolling windows, each predicting two months ahead like Nov-Dec, so a
change is only worth keeping if it lowers the mean clean MAE without a bad window. August is also reported
on its own because its quote_signal behaves the most like Nov-Dec's.

    python experiments.py   # writes results/experiments.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

import train as T

WINDOWS = {  # test window: (start, end). Training uses every clean load before the start.
    "May-Jun": ("2025-05-01", "2025-07-01"),
    "Jul-Aug": ("2025-07-01", "2025-09-01"),
    "Sep-Oct": (T.HOLDOUT_START, "2025-11-01"),
}


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


def run(train, test, feats=T.features, params=None, trend=True, half_life=None, model="lgbm"):
    slope = T.trend_slope(train) if trend else 0.0
    y = T.log_rpm(train) - slope * train["date"].dt.dayofyear.to_numpy()
    if model == "lgbm":
        age = (train["date"].max() - train["date"]).dt.days.to_numpy()
        weight = None if half_life is None else 0.5 ** (age / half_life)
        m = lgb.LGBMRegressor(n_estimators=T.N_TREES, **{**T.PARAMS, **(params or {})})
        m.fit(feats(train), y, sample_weight=weight)
    else:
        m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05,
                                          categorical_features="from_dtype", random_state=0)
        m.fit(feats(train), y)
    pred = m.predict(feats(test)) + slope * test["date"].dt.dayofyear.to_numpy()
    return np.exp(pred) * test["distance"].to_numpy()


def tree_curve(data, counts=(250, 500, 1000, 1500, 2000, 3000)):
    """Clean MAE of the final model by number of trees, read off one 3,000-tree fit per window."""
    rows = {}
    for window, (start, end) in WINDOWS.items():
        train = T.drop_bad_labels(data[data["date"] < start])
        test = data[(data["date"] >= start) & (data["date"] < end)]
        is_clean = test.index.isin(T.drop_bad_labels(test).index)
        model, slope = T.fit(train, max(counts))
        X, t = T.features(test), test["date"].dt.dayofyear.to_numpy()
        for n in counts:
            p = np.exp(model.predict(X, num_iteration=n) + slope * t) * test["distance"].to_numpy()
            rows.setdefault(window, {})[n] = np.abs(test["posted_rate"].to_numpy() - p)[is_clean].mean()
    return pd.DataFrame(rows).T.round(2)


def main():
    raw, valid_raw = pd.read_csv(T.TRAIN_FILE), pd.read_csv(T.VALID_FILE)
    data = T.clean(raw, T.daily_means(raw, valid_raw))
    with_quote = lambda df: T.features(df).assign(quote_signal=df["quote_signal"])
    with_doy = lambda df: T.features(df).assign(dayofyear=df["date"].dt.dayofyear)
    no_quote_daily = lambda df: T.features(df).drop(columns="quote_daily")

    variants = {  # name: keyword arguments for run()
        "final model": {},
        "+ per-load quote_signal": {"feats": with_quote},
        "+ day-of-year feature": {"feats": with_doy},
        "no day's mean quote": {"feats": no_quote_daily},
        "no trend term": {"trend": False},
        "L1 loss instead of Huber": {"params": {"objective": "l1"}},
        "recency weights (60-day half-life)": {"half_life": 60},
        "+ straight-line distance, circuity": {"feats": lane_features},
        "bigger trees (63 leaves)": {"params": {"num_leaves": 63}},
        "sklearn HistGradientBoosting": {"model": "hgb"},
    }

    rows = {name: {} for name in variants}
    for window, (start, end) in WINDOWS.items():
        train = T.drop_bad_labels(data[data["date"] < start])
        test = data[(data["date"] >= start) & (data["date"] < end)]
        is_clean = test.index.isin(T.drop_bad_labels(test).index)
        y = test["posted_rate"].to_numpy()
        for name, kwargs in variants.items():
            p = run(train, test, **kwargs)
            err = np.abs(y - p)
            rows[name][f"MAE {window}"] = err.mean()
            rows[name][f"MAE_clean {window}"] = err[is_clean].mean()
            if window == "Jul-Aug":  # August's quotes look the most like Nov-Dec's
                rows[name]["MAE_clean Aug"] = err[is_clean & (test["date"].dt.month == 8).to_numpy()].mean()
            if window == "Sep-Oct":
                rows[name]["MAPE_% Sep-Oct"] = (err / y).mean() * 100
                rows[name]["within_5% Sep-Oct"] = (err / y < 0.05).mean() * 100
                rows[name]["bias_% Sep-Oct"] = (np.median(p[is_clean] / y[is_clean]) - 1) * 100
        print(f"{window} done, final model MAE {rows['final model'][f'MAE {window}']:.2f}")

    table = pd.DataFrame(rows).T
    clean_cols = [f"MAE_clean {w}" for w in WINDOWS]
    table["MAE_clean mean"] = table[clean_cols].mean(axis=1)
    table["MAE_clean worst"] = table[clean_cols].max(axis=1)
    table = table.round(2)
    T.RESULTS.mkdir(exist_ok=True)
    table.to_csv(T.RESULTS / "experiments.csv")
    print(table.to_string())

    curve = tree_curve(data)
    curve.to_csv(T.RESULTS / "trees.csv")
    print("\nclean MAE by number of trees:\n" + curve.to_string())


if __name__ == "__main__":
    main()
