# %% Imports and paths

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, precision_recall_curve


PROJECT_PATH = Path(r"C:\path\to\CVCML")
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 21 (v0.5 Target Review Framing)"
PLOT_PATH = OUTPUT_PATH / "plots"
PLOT_PATH.mkdir(parents=True, exist_ok=True)

MODEL_COMPARISON_FILE = OUTPUT_PATH / "v0_5_run21_target_framing_model_comparison.csv"
TOPK_ROW_FILE = OUTPUT_PATH / "v0_5_run21_target_framing_topk_row_review.csv"
VALIDATION_PREDICTIONS_FILE = OUTPUT_PATH / "v0_5_run21_target_framing_validation_predictions.csv"

HORIZON_ORDER = {"48h": 48, "72h": 72, "168h_7d": 168}
BEST_FEATURE = "static_labs_vitals_therapy"
COLORS = {
    "static_labs": (31, 119, 180),
    "static_labs_vitals_therapy": (214, 39, 40),
    "48h": (31, 119, 180),
    "72h": (255, 127, 14),
    "168h_7d": (44, 160, 44),
}


# %% Drawing helpers

def load_font(size=28, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_AXIS = load_font(28)
FONT_TICK = load_font(24)
FONT_LEGEND = load_font(24)


def nice_max(value):
    if value <= 0:
        return 1.0
    return float(np.ceil(value * 10) / 10)


def draw_axes(draw, box, x_label, y_label, title, x_ticks, y_max):
    left, top, right, bottom = box
    draw.rectangle((0, 0, 1600, 1000), fill="white")
    draw.text((800, 40), title, font=FONT_TITLE, fill="black", anchor="ma")
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.line((left, top, left, bottom), fill="black", width=3)

    for x in x_ticks:
        px = left + (x - min(x_ticks)) / (max(x_ticks) - min(x_ticks)) * (right - left)
        draw.line((px, bottom, px, bottom + 10), fill="black", width=2)
        draw.text((px, bottom + 18), str(int(x)), font=FONT_TICK, fill="black", anchor="ma")

    for frac in np.linspace(0, 1, 6):
        y_value = y_max * frac
        py = bottom - frac * (bottom - top)
        draw.line((left - 10, py, left, py), fill="black", width=2)
        draw.text((left - 18, py), f"{y_value:.2f}", font=FONT_TICK, fill="black", anchor="rm")
        if frac > 0:
            draw.line((left, py, right, py), fill=(225, 225, 225), width=1)

    draw.text(((left + right) / 2, 940), x_label, font=FONT_AXIS, fill="black", anchor="ma")
    y_img = Image.new("RGBA", (240, 60), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((120, 30), y_label, font=FONT_AXIS, fill="black", anchor="mm")
    y_img = y_img.rotate(90, expand=True)
    return y_img


def scale_point(x, y, box, x_min, x_max, y_min, y_max):
    left, top, right, bottom = box
    px = left + (x - x_min) / (x_max - x_min) * (right - left)
    py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
    return px, py


def draw_line_chart(data, y_col, title, y_label, output_file, legend_col="feature_set"):
    img = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(img)
    box = (180, 150, 1450, 800)
    x_ticks = [48, 72, 168]
    y_max = nice_max(data[y_col].max() * 1.15)
    y_axis = draw_axes(draw, box, "Prediction horizon (hours)", y_label, title, x_ticks, y_max)
    img.paste(y_axis, (35, 390), y_axis)

    for label, sub in data.groupby(legend_col):
        sub = sub.sort_values("horizon_hours")
        color = COLORS.get(label, (80, 80, 80))
        pts = [scale_point(row.horizon_hours, getattr(row, y_col), box, 48, 168, 0, y_max) for row in sub.itertuples()]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=5)
        for px, py in pts:
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color)

    legend_x, legend_y = 980, 150
    for idx, label in enumerate(data[legend_col].unique()):
        color = COLORS.get(label, (80, 80, 80))
        y = legend_y + idx * 36
        draw.line((legend_x, y, legend_x + 44, y), fill=color, width=5)
        draw.text((legend_x + 60, y), str(label), font=FONT_LEGEND, fill="black", anchor="lm")

    img.save(output_file)


def draw_pr_curves(validation_predictions, output_file):
    img = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(img)
    box = (180, 150, 1450, 800)
    y_axis = draw_axes(
        draw,
        box,
        "Recall / sensitivity",
        "Precision / PPV",
        "Run 21 Enriched Model PR Curves by Horizon",
        [0, 0.25, 0.50, 0.75, 1.0],
        1.0,
    )
    img.paste(y_axis, (35, 390), y_axis)

    legend_x, legend_y = 930, 150
    for idx, horizon_label in enumerate(HORIZON_ORDER):
        scored = validation_predictions[
            validation_predictions["feature_set"].eq(BEST_FEATURE)
            & validation_predictions["horizon"].eq(horizon_label)
        ].copy()
        precision, recall, _ = precision_recall_curve(scored["target"].astype(int), scored["platt_probability"])
        ap = average_precision_score(scored["target"].astype(int), scored["platt_probability"])
        color = COLORS[horizon_label]
        pts = [scale_point(float(r), float(p), box, 0, 1, 0, 1) for p, r in zip(precision, recall)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        y = legend_y + idx * 36
        draw.line((legend_x, y, legend_x + 44, y), fill=color, width=5)
        draw.text((legend_x + 60, y), f"{horizon_label} AP={ap:.3f}", font=FONT_LEGEND, fill="black", anchor="lm")

    base_rate = validation_predictions[
        validation_predictions["feature_set"].eq(BEST_FEATURE)
        & validation_predictions["horizon"].eq("168h_7d")
    ]["target"].mean()
    py = scale_point(0, base_rate, box, 0, 1, 0, 1)[1]
    draw.line((box[0], py, box[2], py), fill=(130, 130, 130), width=2)
    draw.text((box[2] - 5, py - 8), f"168h base={base_rate:.3f}", font=FONT_TICK, fill=(80, 80, 80), anchor="ra")

    img.save(output_file)


# %% Load data and create plots

model_comparison = pd.read_csv(MODEL_COMPARISON_FILE)
topk_row_review = pd.read_csv(TOPK_ROW_FILE)
validation_predictions = pd.read_csv(VALIDATION_PREDICTIONS_FILE)

validation_metrics = model_comparison[
    model_comparison["split"].eq("validation")
    & model_comparison["calibration"].eq("platt")
].copy()
validation_metrics["horizon_hours"] = validation_metrics["horizon"].map(HORIZON_ORDER)

draw_line_chart(
    validation_metrics,
    "pr_auc",
    "Run 21 Validation PR-AUC by Horizon",
    "Validation PR-AUC",
    PLOT_PATH / "v0_5_run21_pr_auc_by_horizon.png",
)

draw_line_chart(
    validation_metrics,
    "pr_auc_lift_over_prevalence",
    "Run 21 Validation Lift by Horizon",
    "PR-AUC lift over prevalence",
    PLOT_PATH / "v0_5_run21_pr_auc_lift_by_horizon.png",
)

top5 = topk_row_review[
    topk_row_review["policy"].eq("top_5_percent_rows")
    & topk_row_review["calibration"].eq("platt")
].copy()
top5["horizon_hours"] = top5["horizon"].map(HORIZON_ORDER)
draw_line_chart(
    top5,
    "precision_ppv",
    "Run 21 Top 5% Review Yield",
    "Top 5% row-level PPV",
    PLOT_PATH / "v0_5_run21_top5_ppv_by_horizon.png",
)

draw_pr_curves(
    validation_predictions,
    PLOT_PATH / "v0_5_run21_enriched_pr_curves_by_horizon.png",
)

print(f"Saved Run 21 plots to: {PLOT_PATH}")

