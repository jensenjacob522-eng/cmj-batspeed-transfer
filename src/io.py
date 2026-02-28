import pandas as pd

def load_force_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Handle accidental empty file early with a clear message
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise ValueError(f"CSV looks empty or has no columns: {path}. Make sure it is saved and has a header row.")

    df.columns = [c.strip() for c in df.columns]

    if "time_s" not in df.columns or "force_n" not in df.columns:
        raise ValueError(f"CSV must have columns time_s, force_n. Found: {list(df.columns)}")

    return df[["time_s", "force_n"]].copy()

