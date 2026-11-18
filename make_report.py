"""Build report/report.docx from results/ and the chart made by score.py.

    python make_report.py
"""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches

metrics = json.loads(Path("results/metrics.json").read_text())
experiments = pd.read_csv("results/experiments.csv", index_col=0)
chart = Path("scorer_results/candidate_december.png")

doc = Document()
doc.add_heading("Freight Rate Prediction", 0)


def bullets(lines):
    for line in lines:
        doc.add_paragraph(line, style="List Bullet")


def table(header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = text
    for row in rows:
        for cell, value in zip(t.add_row().cells, row):
            cell.text = str(value)
    doc.add_paragraph()


rows = metrics["rows"]
base, model = metrics["baseline"], metrics["lightgbm"]
final = experiments.loc["final model"]

doc.add_heading("1. Data overview and findings", 1)
bullets([
    "train_test.csv has 48,000 labelled loads from 1 Jan to 31 Oct 2025. validation.csv has 12,000 loads "
    "from 1 Nov to 31 Dec 2025, all later than the training data, so this is a forecasting problem.",
    "Distance drives the price (correlation 0.91 with posted_rate). Rate per mile drops on longer trips, "
    "from $2.54/mi for the shortest sixth of loads to $1.91/mi for the longest. The model therefore "
    "predicts log(rate per mile) instead of the raw rate.",
    "Equipment matters: median Reefer $2.31/mi, Flatbed $2.22/mi, Dry Van $2.05/mi.",
    "market_index varies per load around a daily level (about 155 distinct values per day). Within a "
    "distance band it correlates about 0.3 with rate per mile.",
    "quote_signal looks like a noisy rate-per-mile quote (mean 2.06 vs a median of $2.15/mi) and is much "
    "noisier on short loads. Once distance and equipment are known it adds very little.",
    "Rates drift upward over the year by roughly 0.3% per month after accounting for everything else.",
    "Eight cities in validation never appear in training (e.g. Chicago, Charlotte, San Diego), so cities "
    "are represented by their coordinates rather than by name.",
])

doc.add_heading("2. Data quality issues", 1)
dropped = rows["train"] - rows["train_after_dropping_bad_labels"]
table(["Issue", "Rows", "Handling"], [
    ["Negative weight", "292 train / 145 validation",
     "Absolute value. The magnitudes match normal weights (mean 31.7k lb), so these are sign errors."],
    ["Missing weight", "300 / 165", "Median weight for the equipment type"],
    ["Missing market_index", "374 / 249", "Mean market_index for the same day"],
    ["Bad labels: $/mile below 0.45x or above 2.5x the equipment median",
     f"629 in total, {dropped} in Jan-Aug (1.3%)",
     "Removed from training only. The holdout is scored on every row."],
])
doc.add_paragraph(
    "The bad labels range from 0.16x to 6.3x the normal price with no common factor, so they cannot be "
    "reversed. Example: Baltimore to Oklahoma City by Reefer is priced $3,835 to $4,364 on four dates and "
    "$20,132 on 11 Sep. A classifier trained to spot these rows from the load features scored an AUC of "
    "0.49, i.e. they occur at random and cannot be anticipated in the validation set either.")

doc.add_heading("3. Validation approach and split", 1)
bullets([
    "The split is by time, not random. Validation loads are dated after all labelled data, so a random "
    "split would let the model see future market conditions and overstate accuracy.",
    f"Train: Jan-Aug ({rows['train']:,} loads). Holdout: Sep-Oct ({rows['holdout']:,} loads). Two months "
    "ahead, matching the two-month Nov-Dec horizon.",
    f"The number of trees ({metrics['n_trees']}) is picked by early stopping on August, inside the "
    "training window. The holdout is never used for fitting or tuning.",
    "Holdout scores are reported on all rows and on clean rows (bad labels excluded).",
    "After evaluation the model is refit on all clean Jan-Oct data with the same settings.",
])

doc.add_heading("4. Model", 1)
doc.add_paragraph(
    "LightGBM gradient boosted trees predicting log(rate per mile), with a Huber loss so that the "
    "remaining bad labels have limited influence. Features: distance, log distance, weight, equipment, "
    "market_index, quote_signal, pickup and delivery coordinates, day of week and day of year.")
doc.add_paragraph(
    "Trees cannot extrapolate beyond the last date they were trained on. To carry the upward drift into "
    "Nov-Dec, a linear trend is estimated from out-of-fold residuals of a model without date features, "
    "removed from the target before training and added back when predicting "
    f"({metrics['trend_pct_per_30_days']:+.2f}% per 30 days on the final fit).")

doc.add_heading("5. Results on the Sep-Oct holdout", 1)


def result_row(name, r):
    a, c = r["all_rows"], r["clean_rows"]
    return [name, f"${a['MAE']:.2f}", f"{a['MAPE_%']:.2f}%", f"{a['within_5%']:.1f}%",
            f"${c['MAE']:.2f}", f"{c['MAPE_%']:.2f}%"]


table(["Model", "MAE", "MAPE", "Within 5%", "MAE (clean)", "MAPE (clean)"], [
    result_row("Baseline: median $/mile by equipment x distance", base),
    result_row("LightGBM + trend (final)", model),
])

doc.add_paragraph("What else was tried (one change at a time against the final model):")
table(["Variant", "MAE", "MAPE", "Within 5%", "MAE (clean)", "Bias"], [
    [name, f"${r['MAE']:.2f}", f"{r['MAPE_%']:.2f}%", f"{r['within_5%']:.1f}%",
     f"${r['MAE_clean']:.2f}", f"{r['bias_%']:+.2f}%"]
    for name, r in experiments.iterrows()
])
doc.add_paragraph(
    f"The trend term and the Huber loss are the two changes that helped. Removing the trend raises MAE to "
    f"${experiments.loc['no trend term', 'MAE']:.2f}. The extra distance features gave a gain of "
    f"${final['MAE'] - experiments.loc['+ straight-line distance, circuity', 'MAE']:.2f}, too small to "
    "justify the added complexity. About half of the remaining absolute error comes from the 1.3% of bad "
    "labels; on clean rows the error is around 2%, which is close to the natural spread of prices for the "
    "same lane.")

doc.add_heading("6. December prediction chart", 1)
doc.add_paragraph(
    "Fixed load: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb, only the date changes. The chart "
    "inputs have no market_index or quote_signal, so market_index is set to the daily mean observed in "
    "validation.csv for that date and quote_signal to its training median. Predictions are about "
    "$2.27/mi, in line with the lane's historical Dry Van median of $2.23/mi over 21 loads.")
if chart.exists():
    doc.add_picture(str(chart), width=Inches(6.5))

doc.add_heading("7. Limitations and next steps", 1)
bullets([
    "Bad labels in the validation set, if present, cannot be predicted and will dominate RMSE.",
    "The trend is linear and extrapolated two months. The holdout still showed about 1% "
    "under-prediction, which looks like a real step up in September.",
    "There is no December in the training data, so any holiday-season effect is not learned.",
    "Data that would most likely help: weekly diesel prices, a public spot-rate index, a holiday "
    "calendar and carrier or shipper identifiers.",
])

out = Path("report/report.docx")
out.parent.mkdir(exist_ok=True)
doc.save(out)
print(f"saved {out}")
