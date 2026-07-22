# %% Imports and paths

import os
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    print("Missing characterization dependency:", exc)
    print("")
    print("Install the required packages in your project environment:")
    print("  pip install pandas numpy pillow")
    sys.exit(1)


PROJECT_PATH = Path(r"C:\path\to\CVCML")
STATIC_RUN5_PATH = PROJECT_PATH / "Outputs" / "Run 5 (v0.3a Strict Organism Sensitivity)"
STATIC_RUN6_PATH = PROJECT_PATH / "Outputs" / "Run 6 (Static Model Characterization)"
DYNAMIC_RUN14_PATH = PROJECT_PATH / "Outputs" / "Run 14 (v0.4H Split Dynamic Use Cases)"
OUTPUT_PATH = PROJECT_PATH / "Outputs" / "Run 15 (v0.4I Final Candidate Characterization)"
PLOT_PATH = OUTPUT_PATH / "plots"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH.mkdir(parents=True, exist_ok=True)


# %% Helpers

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def clean_label(text):
    return str(text).replace("_", " ")


def safe_divide(num, den):
    return num / den if den else np.nan


def family_for_feature(name):
    name = str(name)
    if name.startswith("caregiver_"):
        return "Caregiver"
    if name.startswith("linecare_"):
        return "Line care"
    if name.startswith("fluid_") or name.endswith("_output_ml"):
        return "Fluid balance"
    if name.startswith("phys_"):
        return "Derived physiology"
    if name.startswith(("systemic_antibiotic_", "vasopressor_", "therapy_")):
        return "Therapy"
    if any(token in name for token in ["wbc", "platelets", "hemoglobin", "creatinine", "lactate"]):
        return "Labs"
    if any(token in name for token in ["heart_rate", "respiratory_rate", "temperature", "spo2", "sbp", "dbp", "map"]):
        return "Vitals"
    if any(token in name for token in ["cvc_type", "site_known", "anchor_age", "gender", "insurance", "race", "admission_type", "dwell"]):
        return "Static/context"
    return "Other"


def normalize_importance(path, model_role):
    fi = read_csv(path)
    fi["model_role"] = model_role
    fi["feature_family"] = fi["feature"].map(family_for_feature)
    total = fi["importance"].sum()
    fi["importance_share"] = fi["importance"] / total if total else 0.0
    return fi


def top_risk_rows_from_static_top100(top100, n_stays):
    rows = []
    for label, n_select in [
        ("top_1_percent", max(1, int(np.ceil(n_stays * 0.01)))),
        ("top_2_percent", max(1, int(np.ceil(n_stays * 0.02)))),
        ("top_100_available", min(100, len(top100))),
    ]:
        selected = top100.sort_values("predicted_risk", ascending=False).head(n_select)
        positives = int(selected["clabsi"].sum())
        rows.append({
            "model_role": "Static baseline",
            "review_unit": label,
            "flagged": int(len(selected)),
            "true_positive": positives,
            "recall_sensitivity": np.nan,
            "precision_ppv": safe_divide(positives, len(selected)),
            "false_alerts_per_true_positive": safe_divide(len(selected) - positives, positives),
            "note": "Static top-risk recall unavailable from saved top-100-only file.",
        })
    rows.append({
        "model_role": "Static baseline",
        "review_unit": "top_5_percent",
        "flagged": int(np.ceil(n_stays * 0.05)),
        "true_positive": np.nan,
        "recall_sensitivity": np.nan,
        "precision_ppv": np.nan,
        "false_alerts_per_true_positive": np.nan,
        "note": "Requires full static test-set prediction scores; Run 6 saved top 100 only.",
    })
    rows.append({
        "model_role": "Static baseline",
        "review_unit": "top_10_percent",
        "flagged": int(np.ceil(n_stays * 0.10)),
        "true_positive": np.nan,
        "recall_sensitivity": np.nan,
        "precision_ppv": np.nan,
        "false_alerts_per_true_positive": np.nan,
        "note": "Requires full static test-set prediction scores; Run 6 saved top 100 only.",
    })
    return rows


# %% Load selected runs

print("Loading frozen candidate outputs...")

static_summary = read_csv(STATIC_RUN6_PATH / "run6_static_characterization_summary.csv").iloc[0]
static_thresholds = read_csv(STATIC_RUN6_PATH / "run6_alarm_burden_threshold_table.csv")
static_top100 = read_csv(STATIC_RUN6_PATH / "run6_top_100_predicted_risk_stays.csv")
static_cal = read_csv(STATIC_RUN6_PATH / "run6_calibration_deciles.csv")

dynamic_selection = read_csv(DYNAMIC_RUN14_PATH / "v0_4h_split_dynamic_use_case_selection_summary.csv")
dynamic_policy = read_csv(DYNAMIC_RUN14_PATH / "v0_4h_split_dynamic_alert_policy_summary.csv")
dynamic_top = read_csv(DYNAMIC_RUN14_PATH / "v0_4h_split_dynamic_top_risk_review_table.csv")
dynamic_cal = read_csv(DYNAMIC_RUN14_PATH / "v0_4h_split_dynamic_calibration_deciles.csv")


# %% Candidate definitions

static_candidate = {
    "model_role": "Static baseline",
    "source_run": "Run 5/6 - v0.3a strict organism static",
    "model": static_summary["model"],
    "clinical_use": "Baseline CLABSI risk stratification near catheter reference time",
    "prediction_frame": "Single static risk score",
    "label_definition": "Strict organism CLABSI label",
    "horizon_hours": np.nan,
    "score_version": "raw XGBoost probability",
    "roc_auc": static_summary["roc_auc"],
    "pr_auc": static_summary["pr_auc"],
    "brier_score": static_summary["brier_score"],
    "base_rate": static_summary["base_rate"],
    "threshold": static_summary["threshold"],
    "recall_sensitivity": static_summary["recall_sensitivity"],
    "specificity": static_summary["specificity"],
    "precision_ppv": static_summary["precision_ppv"],
    "alerts": static_summary["alerts"],
    "alerts_per_100": static_summary["alerts_per_100_stays"],
    "false_alerts_per_true_positive": static_summary["false_alerts_per_true_positive"],
    "true_positive": static_summary["true_positive"],
    "false_positive": static_summary["false_positive"],
    "false_negative": static_summary["false_negative"],
    "primary_interpretation": "Best overall discriminator; useful for baseline risk, not time-updated surveillance.",
}

selected_dynamic = dynamic_selection[
    (
        dynamic_selection["use_case"].eq("168h_surveillance_review")
        & dynamic_selection["selection_goal"].eq("best_validation_calibration")
    )
    | (
        dynamic_selection["use_case"].eq("72h_near_term_workflow")
        & dynamic_selection["selection_goal"].eq("best_validation_ranking")
    )
].copy()

dynamic_candidates = []
for _, row in selected_dynamic.iterrows():
    role = (
        "168h dynamic surveillance"
        if row["use_case"] == "168h_surveillance_review"
        else "72h dynamic workflow"
    )
    clinical_use = (
        "7-day infection-prevention surveillance and review-list prioritization"
        if row["use_case"] == "168h_surveillance_review"
        else "Near-term workflow-aware monitoring and care-process signal analysis"
    )
    interpretation = (
        "Best dynamic operational candidate; lower alert burden and review-list framing."
        if row["use_case"] == "168h_surveillance_review"
        else "Secondary workflow model; care-process features help but alert burden remains high."
    )
    dynamic_candidates.append({
        "model_role": role,
        "source_run": "Run 14 - v0.4H split dynamic use cases",
        "model": row["model"],
        "clinical_use": clinical_use,
        "prediction_frame": f"Repeated landmark prediction, {int(row['horizon_hours'])}h future window",
        "label_definition": f"{row['label_frame']} strict organism landmark target",
        "horizon_hours": row["horizon_hours"],
        "score_version": row["score_version"],
        "roc_auc": row["roc_auc"],
        "pr_auc": row["pr_auc"],
        "brier_score": row["brier_score"],
        "base_rate": row["base_rate"],
        "threshold": row["threshold"],
        "recall_sensitivity": row["recall_sensitivity"],
        "specificity": row["specificity"],
        "precision_ppv": row["precision_ppv"],
        "alerts": row["alerts"],
        "alerts_per_100": row["alerts_per_100_assessments"],
        "false_alerts_per_true_positive": row["false_alerts_per_true_positive"],
        "true_positive": row["true_positive"],
        "false_positive": row["false_positive"],
        "false_negative": row["false_negative"],
        "primary_interpretation": interpretation,
    })

candidate_summary = pd.DataFrame([static_candidate] + dynamic_candidates)


# %% Clinical role table

clinical_role_summary = pd.DataFrame([
    {
        "model_role": "Static baseline",
        "where_it_lives": "At or near catheter reference time",
        "best_use": "Baseline risk stratification and static literature comparison",
        "not_for": "Repeated bedside alerts or prospective 7-day surveillance",
        "headline_strength": "Strongest discrimination and PR-AUC",
        "main_limitation": "Static timing and residual documentation/context dependence",
    },
    {
        "model_role": "168h dynamic surveillance",
        "where_it_lives": "Daily/periodic infection-prevention review list",
        "best_use": "7-day risk ranking, surveillance, comparison to dynamic CLABSI literature",
        "not_for": "Immediate nurse paging",
        "headline_strength": "Best dynamic operational candidate with lower alert burden",
        "main_limitation": "Lower PR-AUC than static model; still noisy at assessment-level thresholds",
    },
    {
        "model_role": "72h dynamic workflow",
        "where_it_lives": "Near-term monitoring/review experiment",
        "best_use": "Testing care-process, fluid-balance, and workflow-intensity signal",
        "not_for": "Standalone bedside alert deployment",
        "headline_strength": "Care-process features improve 72h dynamic ranking",
        "main_limitation": "Low PPV and high false alerts at high recall",
    },
])


# %% Alert burden comparison

static_alert = static_thresholds.copy()
static_alert["model_role"] = "Static baseline"
static_alert["policy_or_threshold"] = static_alert["threshold"].map(lambda x: f"threshold_{x:.3f}")
static_alert = static_alert[[
    "model_role", "policy_or_threshold", "threshold", "alerts_per_100_stays",
    "recall_sensitivity", "precision_ppv", "false_alerts_per_true_positive",
    "missed_cases",
]]

keys = selected_dynamic[["use_case", "model", "score_version"]].copy()
dyn_policy = dynamic_policy.merge(keys, on=["use_case", "model", "score_version"], how="inner")
dyn_policy["model_role"] = np.where(
    dyn_policy["use_case"].eq("168h_surveillance_review"),
    "168h dynamic surveillance",
    "72h dynamic workflow",
)
dyn_alert = dyn_policy.copy()
dyn_alert["policy_or_threshold"] = dyn_alert["policy"]
dyn_alert = dyn_alert.rename(columns={"alerts_per_100_stays": "alerts_per_100"})
dyn_alert = dyn_alert[[
    "model_role", "policy_or_threshold", "threshold", "alerts_per_100",
    "recall_sensitivity", "precision_ppv", "false_alerts_per_true_positive",
]]

static_alert = static_alert.rename(columns={"alerts_per_100_stays": "alerts_per_100"})
alert_burden_comparison = pd.concat([static_alert, dyn_alert], ignore_index=True, sort=False)


# %% Top-risk comparison

top_risk_rows = top_risk_rows_from_static_top100(static_top100, int(static_summary["n_stays"]))

dyn_top = dynamic_top.merge(keys, on=["use_case", "model", "score_version"], how="inner")
dyn_top = dyn_top[dyn_top["unit"].eq("stay") & dyn_top["top_percent"].isin([1, 5, 10])].copy()
dyn_top["model_role"] = np.where(
    dyn_top["use_case"].eq("168h_surveillance_review"),
    "168h dynamic surveillance",
    "72h dynamic workflow",
)
for _, row in dyn_top.iterrows():
    top_risk_rows.append({
        "model_role": row["model_role"],
        "review_unit": f"top_{int(row['top_percent'])}_percent",
        "flagged": int(row["selected"]),
        "true_positive": np.nan,
        "recall_sensitivity": row["recall_sensitivity"],
        "precision_ppv": row["precision_ppv"],
        "false_alerts_per_true_positive": row["false_alerts_per_true_positive"],
        "note": "Dynamic stay-level top-risk output from Run 14.",
    })

top_risk_comparison = pd.DataFrame(top_risk_rows)


# %% Calibration summary

static_calibration = static_cal.tail(3).copy()
static_calibration["model_role"] = "Static baseline"
static_calibration = static_calibration.rename(
    columns={
        "n_stays": "n",
        "observed_strict_clabsi_rate": "observed_event_rate",
    }
)
static_calibration["calibration_group"] = "highest_static_deciles"

dyn_cal = dynamic_cal.merge(keys, on=["use_case", "model", "score_version"], how="inner")
dyn_cal["model_role"] = np.where(
    dyn_cal["use_case"].eq("168h_surveillance_review"),
    "168h dynamic surveillance",
    "72h dynamic workflow",
)
dyn_calibration = (
    dyn_cal.sort_values(["model_role", "risk_decile"])
    .groupby("model_role", as_index=False)
    .tail(3)
    .copy()
)
dyn_calibration["calibration_group"] = "highest_dynamic_deciles"

calibration_summary = pd.concat([
    static_calibration[[
        "model_role", "risk_decile", "n", "positives", "mean_predicted_risk",
        "observed_event_rate", "calibration_group",
    ]],
    dyn_calibration[[
        "model_role", "risk_decile", "n", "positives", "mean_predicted_risk",
        "observed_event_rate", "calibration_group",
    ]],
], ignore_index=True)


# %% Feature family comparison

feature_files = [
    (
        "Static baseline",
        STATIC_RUN5_PATH / "v0_3a_strict_full_feature_importance.csv",
    ),
    (
        "168h dynamic surveillance",
        DYNAMIC_RUN14_PATH / "gray_zone_excluded_h168_v0_4h_168h_baseline_cleaned_therapy_physiology_feature_importance.csv",
    ),
    (
        "72h dynamic workflow",
        DYNAMIC_RUN14_PATH / "gray_zone_excluded_h72_v0_4h_72h_full_care_process_workflow_feature_importance.csv",
    ),
]

feature_importance = pd.concat(
    [normalize_importance(path, role) for role, path in feature_files],
    ignore_index=True,
)
feature_family_summary = (
    feature_importance
    .groupby(["model_role", "feature_family"], as_index=False)["importance_share"]
    .sum()
    .sort_values(["model_role", "importance_share"], ascending=[True, False])
)
top_features = (
    feature_importance
    .sort_values(["model_role", "importance"], ascending=[True, False])
    .groupby("model_role", as_index=False)
    .head(15)
    .reset_index(drop=True)
)


# %% Save tables

candidate_summary_file = OUTPUT_PATH / "v0_4i_candidate_model_summary.csv"
clinical_role_file = OUTPUT_PATH / "v0_4i_clinical_role_summary.csv"
alert_file = OUTPUT_PATH / "v0_4i_alert_burden_comparison.csv"
top_risk_file = OUTPUT_PATH / "v0_4i_top_risk_review_comparison.csv"
calibration_file = OUTPUT_PATH / "v0_4i_calibration_summary.csv"
feature_family_file = OUTPUT_PATH / "v0_4i_feature_family_summary.csv"
top_features_file = OUTPUT_PATH / "v0_4i_top_features_by_candidate.csv"

candidate_summary.to_csv(candidate_summary_file, index=False)
clinical_role_summary.to_csv(clinical_role_file, index=False)
alert_burden_comparison.to_csv(alert_file, index=False)
top_risk_comparison.to_csv(top_risk_file, index=False)
calibration_summary.to_csv(calibration_file, index=False)
feature_family_summary.to_csv(feature_family_file, index=False)
top_features.to_csv(top_features_file, index=False)


# %% Plots

def load_font(size=22, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped_text(draw, xy, text, font, fill, max_width, line_spacing=4):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x0, y0 = xy
    for line in lines:
        draw.text((x0, y0), line, font=font, fill=fill)
        y0 += font.size + line_spacing
    return y0


def save_line_chart(path, title, y_label, series, labels, y_max):
    width, height = 1200, 760
    margin_left, margin_right, margin_top, margin_bottom = 140, 80, 105, 210
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(34, bold=True)
    label_font = load_font(22)
    small_font = load_font(18)

    draw.text((margin_left, 30), title, font=title_font, fill=(20, 20, 20))
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(40, 40, 40), width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(40, 40, 40), width=2)

    for tick in np.linspace(0, y_max, 5):
        y = plot_bottom - (tick / y_max) * plot_height if y_max else plot_bottom
        draw.line((plot_left - 6, y, plot_left, y), fill=(40, 40, 40), width=2)
        draw.text((25, y - 12), f"{tick:.2f}", font=small_font, fill=(40, 40, 40))

    x_positions = []
    denom = max(1, len(labels) - 1)
    for i, label in enumerate(labels):
        x = plot_left + (i / denom) * plot_width
        x_positions.append(x)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill=(40, 40, 40), width=2)
        draw_wrapped_text(draw, (x - 78, plot_bottom + 18), label, small_font, (40, 40, 40), 160)

    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44)]
    for s_idx, (series_name, values) in enumerate(series.items()):
        pts = []
        for x, value in zip(x_positions, values):
            y = plot_bottom - (float(value) / y_max) * plot_height if y_max else plot_bottom
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=colors[s_idx % len(colors)], width=4)
        for x, y in pts:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[s_idx % len(colors)])
        draw.line((plot_right - 210, plot_top + 12 + 28 * s_idx, plot_right - 170, plot_top + 12 + 28 * s_idx), fill=colors[s_idx % len(colors)], width=4)
        draw.text((plot_right - 160, plot_top + 2 + 28 * s_idx), series_name, font=small_font, fill=(40, 40, 40))

    draw.text((margin_left, height - 45), "Frozen candidate", font=label_font, fill=(20, 20, 20))
    draw.text((20, margin_top - 35), y_label, font=label_font, fill=(20, 20, 20))
    img.save(path)


def heat_color(value, max_value):
    ratio = 0 if max_value == 0 else min(1, max(0, value / max_value))
    red = int(245 * ratio + 240 * (1 - ratio))
    green = int(90 * ratio + 245 * (1 - ratio))
    blue = int(60 * ratio + 255 * (1 - ratio))
    return red, green, blue


def save_heatmap(path, title, pivot):
    width, height = 1250, 900
    margin_left, margin_top = 310, 195
    cell_w, cell_h = 255, 55
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(32, bold=True)
    label_font = load_font(20)
    small_font = load_font(17)

    draw.text((80, 35), title, font=title_font, fill=(20, 20, 20))
    max_value = float(pivot.values.max()) if len(pivot) else 1.0

    for j, col in enumerate(pivot.columns):
        draw_wrapped_text(
            draw,
            (margin_left + j * cell_w + 6, margin_top - 88),
            col,
            small_font,
            (40, 40, 40),
            cell_w - 12,
            line_spacing=2,
        )

    for i, row_name in enumerate(pivot.index):
        y = margin_top + i * cell_h
        draw.text((35, y + 16), row_name, font=label_font, fill=(40, 40, 40))
        for j, col in enumerate(pivot.columns):
            x = margin_left + j * cell_w
            value = float(pivot.loc[row_name, col])
            color = heat_color(value, max_value)
            draw.rectangle((x, y, x + cell_w - 6, y + cell_h - 6), fill=color, outline=(220, 220, 220))
            draw.text((x + 88, y + 15), f"{value:.2f}", font=small_font, fill=(25, 25, 25))

    img.save(path)


plot_labels = candidate_summary["model_role"].tolist()
performance_plot = PLOT_PATH / "v0_4i_candidate_model_performance.png"
save_line_chart(
    performance_plot,
    "v0.4I Frozen Candidate Model Performance",
    "Metric",
    {
        "ROC-AUC": candidate_summary["roc_auc"].tolist(),
        "PR-AUC": candidate_summary["pr_auc"].tolist(),
    },
    plot_labels,
    max(0.85, candidate_summary[["roc_auc", "pr_auc"]].max().max() + 0.05),
)

alert_plot = PLOT_PATH / "v0_4i_candidate_alert_burden.png"
save_line_chart(
    alert_plot,
    "v0.4I Alert Burden at Selected Operating Point",
    "False alerts / TP",
    {
        "False alerts / TP": candidate_summary["false_alerts_per_true_positive"].tolist(),
    },
    plot_labels,
    candidate_summary["false_alerts_per_true_positive"].max() + 5,
)

family_pivot = (
    feature_family_summary
    .pivot(index="feature_family", columns="model_role", values="importance_share")
    .fillna(0)
)
feature_plot = PLOT_PATH / "v0_4i_feature_family_heatmap.png"
save_heatmap(
    feature_plot,
    "v0.4I Feature Family Importance by Frozen Candidate",
    family_pivot,
)


# %% Manifest and console summary

manifest_rows = [
    ("Candidate model summary", candidate_summary_file),
    ("Clinical role summary", clinical_role_file),
    ("Alert burden comparison", alert_file),
    ("Top-risk review comparison", top_risk_file),
    ("Calibration summary", calibration_file),
    ("Feature family summary", feature_family_file),
    ("Top features by candidate", top_features_file),
    ("Candidate model performance plot", performance_plot),
    ("Candidate alert burden plot", alert_plot),
    ("Feature family heatmap", feature_plot),
]
manifest_file = OUTPUT_PATH / "v0_4i_final_candidate_characterization_output_manifest.csv"
pd.DataFrame(manifest_rows, columns=["output", "path"]).to_csv(manifest_file, index=False)

print("")
print("v0.4I frozen candidate model summary:")
display_cols = [
    "model_role",
    "clinical_use",
    "roc_auc",
    "pr_auc",
    "recall_sensitivity",
    "precision_ppv",
    "alerts_per_100",
    "false_alerts_per_true_positive",
    "primary_interpretation",
]
print(candidate_summary[display_cols].round(4).to_string(index=False))

print("")
print("Saved outputs:")
for label, path in manifest_rows:
    print(f"  {label}: {path}")

print("")
print("Modeling 12 v0.4I Final Candidate Characterization complete.")

