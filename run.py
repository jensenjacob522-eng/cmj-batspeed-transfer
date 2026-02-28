import argparse
from pathlib import Path

from src.io import load_force_csv
from src.metrics import compute_cmj_metrics
from src.plotting import save_force_time_plot


LABELS = {
    "bw_n": "Bodyweight Estimate (N)",
    "peak_force_n": "Peak Force (N)",
    "peak_force_xbw": "Peak Force (×BW)",
    "net_peak_force_n": "Net Peak Force (N)",
    "time_to_peak_ms": "Time to Peak (ms)",
    "rfd_0_50_n_per_s": "RFD 0–50 ms (N/s)",
    "rfd_0_100_n_per_s": "RFD 0–100 ms (N/s)",
    "rfd_0_200_n_per_s": "RFD 0–200 ms (N/s)",
    "impulse_0_200_ns": "Impulse 0–200 ms (N·s)",
    "net_impulse_0_200_ns": "Net Impulse 0–200 ms (N·s)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--athlete", default="Athlete")
    p.add_argument("--sampling_rate", type=int, required=True)
    args = p.parse_args()

    df = load_force_csv(args.input)
    metrics = compute_cmj_metrics(df, args.sampling_rate)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    fig_path = out_dir / "figures" / f"{args.athlete.replace(' ', '_')}_force_time.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{args.athlete.replace(' ', '_')}_report.html"

    # Save plot
    save_force_time_plot(df, str(fig_path), f"{args.athlete} Force-Time")

    # Build table rows with clean labels
    rows = "\n".join(
        [
            f"<tr><td>{LABELS.get(k, k)}</td><td>{v:.2f}</td></tr>"
            for k, v in metrics.items()
        ]
    )

    # HTML is in /reports, so image path must be relative to that folder
    img_src = f"figures/{fig_path.name}"

    html = f"""
    <html>
      <body style="font-family:Arial;padding:20px">
        <h2>{args.athlete} — CMJ Report</h2>
        <p><b>File:</b> {args.input} | <b>Fs:</b> {args.sampling_rate} Hz</p>
        <table border="0" cellpadding="6">{rows}</table>
        <img src="{img_src}" style="max-width:900px;margin-top:12px"/>
      </body>
    </html>
    """

    html_path.write_text(html, encoding="utf-8")

    print("=== CMJ Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    print(f"\nSaved: {html_path}")


if __name__ == "__main__":
    main()
