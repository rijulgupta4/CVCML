import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


FONT = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def canvas(title, width=1800, height=1100):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 42), title, fill="#222222", font=font(48, True), anchor="ma")
    return image, draw


def axes(draw, left, top, right, bottom, xlabel, ylabel):
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.text(((left + right) // 2, bottom + 75), xlabel, fill="#222222", font=font(30), anchor="ma")
    draw.text((45, (top + bottom) // 2), ylabel, fill="#222222", font=font(30), anchor="mm")


def window_plot(output):
    df = pd.read_csv(output / "run32_secondary_source_window_sensitivity.csv")
    image, draw = canvas("ARMD-MGB Secondary-Source Window Sensitivity")
    left, top, right, bottom = 210, 150, 1700, 900
    axes(draw, left, top, right, bottom, "Secondary-source search window (+/- days)", "")
    draw.text((left, 115), "Proportion with same-organism nonblood culture", fill="#444444",
              font=font(24), anchor="la")
    ymax = max(0.28, df.ci_high.max() * 1.18)
    for tick in range(0, 8):
        value = ymax * tick / 7
        y = bottom - (bottom - top) * tick / 7
        draw.line((left, y, right, y), fill="#E5E7EB", width=2)
        draw.text((left - 22, y), f"{value:.0%}", fill="#444444", font=font(24), anchor="rm")
    points = []
    for row in df.itertuples():
        x = left + (right - left) * (row.window_days - 1) / 2
        y = bottom - (bottom - top) * row.proportion / ymax
        ylo = bottom - (bottom - top) * row.ci_low / ymax
        yhi = bottom - (bottom - top) * row.ci_high / ymax
        draw.line((x, ylo, x, yhi), fill="#176B87", width=5)
        draw.line((x - 15, ylo, x + 15, ylo), fill="#176B87", width=4)
        draw.line((x - 15, yhi, x + 15, yhi), fill="#176B87", width=4)
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#176B87")
        draw.text((x, bottom + 25), str(row.window_days), fill="#333333", font=font(26), anchor="ma")
        draw.text((x, y - 32), f"{row.proportion:.1%}", fill="#176B87", font=font(24, True), anchor="ms")
        points.append((x, y))
    draw.line(points, fill="#176B87", width=5)
    image.save(output / "plots" / "run32_secondary_source_window_sensitivity.png")


def organism_plot(output):
    df = pd.read_csv(output / "run32_mimic_armd_organism_comparison.csv")
    df = df.nlargest(12, "mimic_episodes").sort_values("mimic_share")
    image, draw = canvas("Strict-Rule Organism Profile: MIMIC-IV vs ARMD-MGB", 2000, 1250)
    left, top, right, bottom = 620, 150, 1880, 1040
    axes(draw, left, top, right, bottom, "Share of qualifying organism events", "")
    xmax = max(df.mimic_share.max(), df.armd_share.max()) * 1.15
    row_h = (bottom - top) / len(df)
    for i, row in enumerate(df.itertuples()):
        y = bottom - (i + 0.5) * row_h
        draw.text((left - 20, y), row.organism_norm, fill="#333333", font=font(22), anchor="rm")
        m_end = left + (right - left) * row.mimic_share / xmax
        a_end = left + (right - left) * row.armd_share / xmax
        draw.rectangle((left, y - 24, m_end, y - 4), fill="#3B82B9")
        draw.rectangle((left, y + 4, a_end, y + 24), fill="#D9772B")
    for i in range(6):
        value = xmax * i / 5
        x = left + (right - left) * i / 5
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=2)
        draw.text((x, bottom + 25), f"{value:.0%}", fill="#444444", font=font(22), anchor="ma")
    draw.rectangle((1420, 85, 1460, 105), fill="#3B82B9")
    draw.text((1475, 95), "MIMIC-IV episodes", fill="#333333", font=font(23), anchor="lm")
    draw.rectangle((1690, 85, 1730, 105), fill="#D9772B")
    draw.text((1745, 95), "ARMD-MGB accessions", fill="#333333", font=font(23), anchor="lm")
    image.save(output / "plots" / "run32_mimic_armd_organism_profile.png")


def subgroup_plot(output):
    df = pd.read_csv(output / "run32_armd_subgroup_stability.csv")
    df = df[df.variable.isin(["gender", "setting"]) & (df.positive_blood_accessions >= 100)].copy()
    df["label"] = df.variable + ": " + df.level.astype(str)
    df = df.sort_values("strict_rate")
    image, draw = canvas("ARMD-MGB Label Stability by Setting and Sex")
    left, top, right, bottom = 450, 150, 1700, 900
    axes(draw, left, top, right, bottom, "Strict-rule positivity among positive blood cultures", "")
    xmin = max(0, df.strict_ci_low.min() - 0.03)
    xmax = min(1, df.strict_ci_high.max() + 0.03)
    row_h = (bottom - top) / len(df)
    for i, row in enumerate(df.itertuples()):
        y = bottom - (i + 0.5) * row_h
        scale = lambda v: left + (right - left) * (v - xmin) / (xmax - xmin)
        draw.text((left - 20, y), row.label, fill="#333333", font=font(25), anchor="rm")
        draw.line((scale(row.strict_ci_low), y, scale(row.strict_ci_high), y), fill="#2A7F62", width=5)
        x = scale(row.strict_rate)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#2A7F62")
    for i in range(6):
        value = xmin + (xmax - xmin) * i / 5
        x = left + (right - left) * i / 5
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=2)
        draw.text((x, bottom + 25), f"{value:.0%}", fill="#444444", font=font(22), anchor="ma")
    image.save(output / "plots" / "run32_subgroup_strict_rule_stability.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (args.output / "plots").mkdir(exist_ok=True)
    window_plot(args.output)
    organism_plot(args.output)
    subgroup_plot(args.output)


if __name__ == "__main__":
    main()

"""Render aggregate Run 32 transportability figures.

Input tables remain local because they are derived from credentialed data.
"""
