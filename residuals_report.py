import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_column(df: pd.DataFrame, preferred: list[str], must_contain: list[str], avoid_contains: list[str] | None = None) -> str:
    """
    1) If any preferred column exists EXACTLY, use it.
    2) Else, find a column containing all must_contain and none of avoid_contains.
    """
    avoid_contains = avoid_contains or []
    cols = list(df.columns)

    # 1) exact match preference
    for name in preferred:
        if name in df.columns:
            return name

    # 2) fallback search
    for c in cols:
        lc = c.lower()
        if all(s.lower() in lc for s in must_contain) and not any(bad.lower() in lc for bad in avoid_contains):
            return c

    raise ValueError(
        f"Could not pick column. preferred={preferred}, must_contain={must_contain}, avoid_contains={avoid_contains}. "
        f"Example cols: {cols[:35]}"
    )


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV file, e.g. data/hp_obp.csv")
    ap.add_argument("--level", default="All", help='High School / College / Pro / All')
    ap.add_argument("--top_n", type=int, default=15)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Force the REAL numeric columns if they exist
    jump_col = pick_column(
        df,
        preferred=["jump_height_(imp-mom)_[cm]_mean_cmj"],
        must_contain=["jump_height", "mean_cmj"],
        avoid_contains=["group"]
    )

    bat_col = pick_column(
        df,
        preferred=["bat_speed_mph"],
        must_contain=["bat_speed_mph"],
        avoid_contains=["group"]  # avoids bat_speed_mph_group
    )

    level_col = "playing_level" if "playing_level" in df.columns else None

    # Optional filter by playing level
    if args.level.lower() != "all" and level_col is not None:
        df = df[df[level_col].astype(str).str.lower() == args.level.lower()].copy()

    # Convert to numeric + drop missing
    df[jump_col] = pd.to_numeric(df[jump_col], errors="coerce")
    df[bat_col] = pd.to_numeric(df[bat_col], errors="coerce")
    df = df.dropna(subset=[jump_col, bat_col]).copy()

    if len(df) < 20:
        raise ValueError(
            f"Not enough valid rows after cleaning. Rows={len(df)}\n"
            f"jump_col={jump_col}\n"
            f"bat_col={bat_col}"
        )

    x = df[jump_col].to_numpy(float)
    y = df[bat_col].to_numpy(float)

    # Fit on ALL rows (best for residual ranking)
    m, b = fit_line(x, y)
    y_pred = m * x + b
    residual = y - y_pred

    df_out = pd.DataFrame({
        "playing_level": df[level_col].astype(str).values if level_col else [""] * len(df),
        "cmj_jump_height_cm": x,
        "actual_bat_speed_mph": y,
        "pred_bat_speed_mph": y_pred,
        "residual_mph": residual,  # + = overperformer, - = underperformer
    })

    # Add athlete id if present
    athlete_id_col = None
    for candidate in ["athlete_uid", "athlete", "name", "player", "id"]:
        if candidate in df.columns:
            athlete_id_col = candidate
            break
    if athlete_id_col:
        df_out.insert(0, athlete_id_col, df[athlete_id_col].astype(str).values)

    top_over = df_out.sort_values("residual_mph", ascending=False).head(args.top_n)
    top_under = df_out.sort_values("residual_mph", ascending=True).head(args.top_n)

    out_dir = Path("reports")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = args.level.replace(" ", "_")
    csv_path = out_dir / f"residuals_{tag}.csv"
    html_path = out_dir / f"residuals_{tag}.html"
    fig_path = fig_dir / f"residuals_scatter_{tag}.png"

    df_out.to_csv(csv_path, index=False)

    plt.figure()
    plt.scatter(df_out["cmj_jump_height_cm"], df_out["actual_bat_speed_mph"], label="Actual")
    x_line = np.linspace(df_out["cmj_jump_height_cm"].min(), df_out["cmj_jump_height_cm"].max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, label="Model Fit")
    plt.xlabel("CMJ Jump Height (cm)")
    plt.ylabel("Bat Speed (mph)")
    plt.title(f"Residuals: Actual − Predicted ({args.level})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    def to_html_table(d: pd.DataFrame) -> str:
        return d.to_html(index=False, float_format=lambda v: f"{v:.2f}")

    html = f"""
    <html>
      <body style="font-family:Arial;padding:20px">
        <h2>Residual Report: CMJ → Bat Speed</h2>
        <p><b>File:</b> {args.input} &nbsp; | &nbsp; <b>Filter:</b> {args.level}</p>
        <p><b>Columns used:</b> CMJ = <code>{jump_col}</code> | Bat = <code>{bat_col}</code></p>
        <p><b>Model:</b> bat_speed = {m:.4f} * cmj + {b:.4f}</p>

        <h3>Plot</h3>
        <img src="figures/{fig_path.name}" style="max-width:900px;margin-top:10px"/>

        <h3>Top Overperformers (Actual > Predicted)</h3>
        {to_html_table(top_over)}

        <h3>Top Underperformers (Actual < Predicted)</h3>
        {to_html_table(top_under)}

        <p style="margin-top:16px;color:#666">
          Residual = Actual Bat Speed − Predicted Bat Speed. Positive = overperformer, Negative = underperformer.
        </p>
      </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")

    print("\n=== Residual Report ===")
    print(f"Jump column: {jump_col}")
    print(f"Bat column:  {bat_col}")
    print(f"Filter:      {args.level}")
    print(f"Model:       bat_speed = {m:.4f} * cmj + {b:.4f}")
    print(f"Rows used:   {len(df_out)}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved plot:  {fig_path}")
    print(f"Saved HTML:  {html_path}\n")


if __name__ == "__main__":
    main()