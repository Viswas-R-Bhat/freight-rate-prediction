# Freight Rate Prediction

Predicting the posted rate of a truckload from its lane, distance, equipment, weight, date and market signals.

The model is LightGBM on log(rate per mile) with a linear time trend and the day's average quote_signal as a market-level feature. It is checked on a time-based holdout (train on Jan-Aug 2025, test on Sep-Oct 2025) and backtested on two earlier windows, so the choices don't overfit one holdout.

## Results

Holdout: 9,523 loads from Sep-Oct 2025, never seen during training.

| Model | MAE | MAPE | Within 5% |
|---|---|---|---|
| Baseline (median $/mile per equipment x distance) | $229.14 | 10.49% | 38.4% |
| **LightGBM + trend + daily quote** | **$95.08** | **4.19%** | **96.7%** |

About 1.3% of the labels are clearly wrong (0.16x to 6.3x the normal price) and make up about half of the MAE. Excluding them, MAE is $43.41 (1.76%).

The same pipeline, retrained on earlier data and predicting two months ahead each time:

| Test window | Trained on | MAE | MAE (clean rows) |
|---|---|---|---|
| May-Jun | Jan-Apr | $97.71 | $49.00 |
| Jul-Aug | Jan-Jun | $94.48 | $44.64 |
| Sep-Oct | Jan-Aug | $95.08 | $43.41 |

![December chart](scorer_results/candidate_december.png)

## Setup

```bash
pip install -r requirements.txt
```

Put the provided data files in `data/` (they are not committed):

```
data/
  train_test.csv
  validation.csv
  december-chart-inputs.csv   # committed, filled with predictions
```

## Running

```bash
python train.py          # holdout metrics, validation_predictions.csv, December predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
python experiments.py    # optional (~2 min), ablations on all three windows and the tree-count curve
```

`python train.py --eval-only` prints the holdout metrics without writing any files.

## Layout

```
train.py                     cleaning, features, holdout evaluation, final fit, predictions
experiments.py               ablations behind the modelling choices, on three backtest windows
score.py                     provided scorer, validates outputs and draws the December chart
validation_predictions.csv   final predictions for the 12,000 validation loads
results/                     metrics.json, experiments.csv, trees.csv
report/report.docx           write-up
scorer_results/              chart from score.py
```

## Approach

**Cleaning**
- Negative weights are sign errors (same magnitudes as normal weights), so they're made positive.
- Missing weight is filled with the equipment median.
- Missing market_index is filled with that day's mean.
- Loads with $/mile below 0.45x or above 2.5x the equipment median are bad labels and are dropped from training. They are kept when scoring.
- 48 more bad labels sit inside that band: all are over 50% off an out-of-fold prediction, and no load is between 10% and 50% off. Dropping them made no measurable difference, so they're left to the Huber loss.
- Weight is capped at 47,500 lb (1,204 training loads) and floored at 5,000 lb, and distance is floored at 70 miles (48 loads). These loads are kept, because their prices are consistent with the capped values.
- quote_signal is within about 1% of the true $/mile for most loads in Jan-Mar, Jun and Sep, but noisy and 4-11% low in Apr-May, Jul-Aug and Oct. Nov-Dec look like the noisy months, and most like August: the spread of quote vs prediction, which needs no labels, is about 0.17 there against 0.015 in the clean months.

**Validation**
- Split by date, because the validation loads all come after the labelled data. A random split would leak future prices.
- Sep-Oct is the holdout, never used for fitting.
- Every modelling choice is also backtested on May-Jun and Jul-Aug (`experiments.py`). A change is kept only if it lowers the mean clean MAE over the three windows without a bad window.
- Results are stable across random seeds: Nov-Dec predictions move about 0.2% between seeds.
- Hiding 8 training cities from the model raised MAPE on their loads only from about 1.75% to 1.9%. So the 8 cities that only appear in validation (12% of its loads), placed by their coordinates, should be fine.

**Model**
- Target is log(rate per mile), which puts short and long loads on the same scale.
- Huber loss limits the pull of the remaining bad labels.
- The day's average quote_signal (over its 150-200 loads) is a feature. Averaging cancels most of the per-load noise, so it tracks the day's market level. Like the market_index fill, it uses features only, no labels. Without it the mean clean MAE over the three windows rises from $45.68 to $56.12. It does cost about $4 in August alone.
- Each load's own quote_signal is not used. It helps in most months but blows up in August ($118.05 vs $50.17 clean MAE), the month whose quotes look most like Nov-Dec.
- dayofyear is not used. Trees can't extrapolate it past October, and adding it back loses badly on Jul-Aug ($71.30 vs $44.64).
- Trees can't extrapolate past October either. So a linear drift (about +0.6% per month) is estimated on the training data, taken out of the target before fitting and added back at prediction time. The drift is estimated without the daily quote, which identifies the date and would absorb it. Without the trend the mean clean MAE is $62.50.
- A fixed 1,500 trees. In every window the error flattens after about 1,000 trees and doesn't rise up to 3,000 (`results/trees.csv`), so more trees don't overfit. Early stopping on one month picked anywhere from 228 to 1,427 trees, and stopping near 250 costs about $4 on Sep-Oct.
- L1 loss, recency weights, extra distance features, bigger trees and sklearn's HistGradientBoosting are all within noise of the final model, so the simpler setup was kept.

**December chart**
- The chart inputs have no market_index or quote_signal. market_index and the daily quote are set to the daily means seen in validation.csv. Predictions are about $2.34/mi, in line with comparable Dry Van loads (300-420 miles) in Sep-Oct.
