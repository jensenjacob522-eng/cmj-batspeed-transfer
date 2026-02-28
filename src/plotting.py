from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_force_time_plot(df: pd.DataFrame, out_path: str, title: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df["time_s"], df["force_n"])
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path
