import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helpers ---

def fit_line(x, y):
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)

def filter_level(df_level: pd.DataFrame, jump_col: str, bat_col: str, min_bat: float = 40.0, z_cut: float = 3.0):
    """Filters: bat >= min_bat and outliers within +/- z_cut SD for both CMJ and bat speed."""
    df = df_level.copy()
    df[jump_col] = pd.to_numeric(df[jump_col], errors="coerce")
    df[bat_col] = pd.to_numeric(df[bat_col], errors="coerce")
    df = df.dropna(subset=[jump_col, bat_col])

    # bat speed cutoff
    df = df[df[bat_col] >= min_bat].copy()
    if len(df) < 10:
        return df

    j = df[jump_col].to_numpy(float)
    b = df[bat_col].to_numpy(float)

    j_mu, j_sd = float(np.mean(j)), float(np.std(j, ddof=0))
    b_mu, b_sd = float(np.mean(b)), float(np.std(b, ddof=0))

    if j_sd == 0 or b_sd == 0:
        return df

    j_z = (j - j_mu) / j_sd
    b_z = (b - b_mu) / b_sd
    keep = (np.abs(j_z) <= z_cut) & (np.abs(b_z) <= z_cut)

    return df.loc[keep].copy()

def analyze_level(df_level: pd.DataFrame, jump_col: str, bat_col: str):
    """Fit regression, compute residual extremes."""
    if len(df_level) < 20:
        return None

    x = df_level[jump_col].to_numpy(float)
    y = df_level[bat_col].to_numpy(float)

    m, b = fit_line(x, y)
    r = float(np.corrcoef(x, y)[0, 1])

    y_pred = m * x + b
    residual = y - y_pred

    uid = df_level["athlete_uid"].astype(str).values if "athlete_uid" in df_level.columns else np.array([""] * len(df_level))

    df_out = pd.DataFrame({
        "athlete_uid": uid,
        "cmj": x,
        "actual": y,
        "predicted": y_pred,
        "residual": residual
    })

    top_over = df_out.sort_values("residual", ascending=False).iloc[0]
    top_under = df_out.sort_values("residual", ascending=True).iloc[0]

    return {
        "rows": len(df_out),
        "m": m,
        "b": b,
        "r": r,
        "top_over": top_over,
        "top_under": top_under,
        "x": x,
        "y": y
    }

# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--min_bat", type=float, default=40.0)
    ap.add_argument("--z_cut", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Your real numeric columns:
    jump_col = "jump_height_(imp-mom)_[cm]_mean_cmj"
    bat_col = "bat_speed_mph"

    if "playing_level" not in df.columns:
        raise ValueError("CSV missing 'playing_level' column.")
    if jump_col not in df.columns:
        raise ValueError(f"CSV missing CMJ column: {jump_col}")
    if bat_col not in df.columns:
        raise ValueError(f"CSV missing bat speed column: {bat_col}")

    levels = ["High School", "College", "Pro"]

    out_dir = Path("reports")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each level after filtering
    results = {}
    counts = []

    for lvl in levels:
        raw = df[df["playing_level"] == lvl].copy()
        raw_n = len(raw)
        filt = filter_level(raw, jump_col, bat_col, min_bat=args.min_bat, z_cut=args.z_cut)
        filt_n = len(filt)
        counts.append({"playing_level": lvl, "rows_raw": raw_n, "rows_filtered": filt_n})

        res = analyze_level(filt, jump_col, bat_col)
        if res is not None:
            results[lvl] = res

    # Save counts CSV
    counts_csv = out_dir / "driveline_transfer_filter_counts.csv"
    pd.DataFrame(counts).to_csv(counts_csv, index=False)

    # --- Overlay plot with regression lines ---
    fig_path = fig_dir / "cmj_batspeed_by_level.png"

    plt.figure()
    # Scatter each group + line
    all_x = []
    for lvl in levels:
        if lvl not in results:
            continue
        x = results[lvl]["x"]
        y = results[lvl]["y"]
        all_x.append(x)
        plt.scatter(x, y, label=f"{lvl} (n={results[lvl]['rows']})")

    # Use global x-range for lines
    if len(all_x) == 0:
        raise ValueError("No levels had enough data after filtering to plot.")
    x_min = float(np.min(np.concatenate(all_x)))
    x_max = float(np.max(np.concatenate(all_x)))
    x_line = np.linspace(x_min, x_max, 200)

    for lvl in levels:
        if lvl not in results:
            continue
        m = results[lvl]["m"]
        b = results[lvl]["b"]
        y_line = m * x_line + b
        plt.plot(x_line, y_line, label=f"{lvl} fit (r={results[lvl]['r']:.2f})")

    plt.xlabel("CMJ Jump Height (cm)")
    plt.ylabel("Bat Speed (mph)")
    plt.title("CMJ → Bat Speed (Filtered) | Regression Line per Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    # --- HTML report ---
    html_path = out_dir / "driveline_transfer_report.html"

    def block(lvl, res):
        over = res["top_over"]
        under = res["top_under"]
        return f"""
        <h2>{lvl}</h2>
        <p><b>Rows used:</b> {res['rows']}</p>
        <p><b>Model:</b> bat_speed = {res['m']:.4f} * cmj + {res['b']:.4f}</p>
        <p><b>Correlation (r):</b> {res['r']:.3f}</p>
        <p><b>Top overperformer:</b> {over['athlete_uid']} | CMJ {over['cmj']:.2f} | Actual {over['actual']:.2f} | Pred {over['predicted']:.2f} | <b>+{over['residual']:.2f}</b></p>
        <p><b>Top underperformer:</b> {under['athlete_uid']} | CMJ {under['cmj']:.2f} | Actual {under['actual']:.2f} | Pred {under['predicted']:.2f} | <b>{under['residual']:.2f}</b></p>
        <hr>
        """

    sections = ""
    for lvl in levels:
        if lvl in results:
            sections += block(lvl, results[lvl])
        else:
            sections += f"<h2>{lvl}</h2><p><i>Not enough data after filtering.</i></p><hr>"

    html = f"""
    <html>
    <body style="font-family:Arial;padding:28px">
      <h1>CMJ → Bat Speed Transfer Efficiency Report (Filtered)</h1>

      <p><b>Input:</b> {args.input}</p>
      <p><b>Filters:</b> bat ≥ {args.min_bat:.1f} mph and outliers within ±{args.z_cut:.1f} SD (CMJ + bat speed)</p>
      <p><b>Counts file:</b> {counts_csv.as_posix()}</p>

      <h2>Overlay Plot</h2>
      <img src="figures/{fig_path.name}" style="max-width:950px;margin-top:8px"/>

      <h2>Level Summaries</h2>
      {sections}

      <p style="color:#666;margin-top:12px">
        Residual = Actual − Predicted. Positive = overperformer, Negative = underperformer.
      </p>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")

    # --- PDF report (ReportLab) ---
    pdf_path = out_dir / "Driveline_CMJ_BatSpeed_Transfer_Report.pdf"
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        y = height - 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "CMJ → Bat Speed Transfer Efficiency Report (Filtered)")
        y -= 18

        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Input: {args.input}")
        y -= 14
        c.drawString(40, y, f"Filters: bat ≥ {args.min_bat:.1f} mph; outliers within ±{args.z_cut:.1f} SD")
        y -= 18

        # Add overlay plot image
        img = ImageReader(str(fig_path))
        img_w = width - 80
        img_h = img_w * 0.62  # good aspect for letter
        c.drawImage(img, 40, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        y = y - img_h - 18

        # Level summaries
        for lvl in levels:
            if y < 120:
                c.showPage()
                y = height - 40

            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, lvl)
            y -= 14

            if lvl not in results:
                c.setFont("Helvetica", 10)
                c.drawString(50, y, "Not enough data after filtering.")
                y -= 18
                continue

            res = results[lvl]
            over = res["top_over"]
            under = res["top_under"]

            c.setFont("Helvetica", 10)
            c.drawString(50, y, f"Rows: {res['rows']} | r = {res['r']:.3f}")
            y -= 12
            c.drawString(50, y, f"Model: bat_speed = {res['m']:.4f} * cmj + {res['b']:.4f}")
            y -= 14
            c.drawString(50, y, f"Top over:  {over['athlete_uid']} | CMJ {over['cmj']:.2f} | Actual {over['actual']:.2f} | Pred {over['predicted']:.2f} | +{over['residual']:.2f}")
            y -= 12
            c.drawString(50, y, f"Top under: {under['athlete_uid']} | CMJ {under['cmj']:.2f} | Actual {under['actual']:.2f} | Pred {under['predicted']:.2f} | {under['residual']:.2f}")
            y -= 18

        c.save()

    except ImportError:
        pdf_path = None

    print("\n=== Driveline Transfer Report Saved ===")
    print(f"Overlay plot:  {fig_path}")
    print(f"HTML report:   {html_path}")
    if pdf_path:
        print(f"PDF report:    {pdf_path}\n")
    else:
        print("PDF not generated because reportlab is not installed.")
        print("Install with: pip install reportlab")
        print("Or open the HTML and Cmd+P → Save as PDF.\n")


if __name__ == "__main__":
    main()