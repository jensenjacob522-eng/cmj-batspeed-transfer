import numpy as np
import pandas as pd


def _force_at_time(time_s: np.ndarray, force: np.ndarray, t: float) -> float:
    """Return force at time t using nearest available sample."""
    idx = int(np.argmin(np.abs(time_s - t)))
    return float(force[idx])


def _impulse_trapz(t: np.ndarray, f: np.ndarray) -> float:
    """Manual trapezoidal integration: sum((f[i]+f[i-1])/2 * dt)."""
    imp = 0.0
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        imp += float((f[i] + f[i - 1]) / 2.0) * dt
    return float(imp)


def compute_cmj_metrics(df: pd.DataFrame, sampling_rate: int) -> dict:
    force = df["force_n"].to_numpy(dtype=float)
    time_s = df["time_s"].to_numpy(dtype=float)

    if len(force) < 3:
        raise ValueError("Not enough rows in file.")

    # Use first timestamp as start
    t0 = float(time_s[0])

    # -----------------------
    # 1) Estimate Bodyweight
    # -----------------------
    # Use first 0.25s if available, otherwise use first 10% of samples (min 5 samples)
    bw_window_s = 0.25
    mask_bw = time_s <= (t0 + bw_window_s)

    if np.sum(mask_bw) < 5:
        n = max(5, int(len(force) * 0.10))
        n = min(n, len(force))
        bw_n = float(np.mean(force[:n]))
    else:
        bw_n = float(np.mean(force[mask_bw]))

    # -----------------------
    # 2) Basic metrics
    # -----------------------
    peak_force = float(np.max(force))
    peak_i = int(np.argmax(force))
    time_to_peak_ms = float((time_s[peak_i] - t0) * 1000.0)  # relative to t0

    # Early force samples for RFD (absolute force)
    f0 = float(force[0])
    f_50 = _force_at_time(time_s, force, t0 + 0.05)
    f_100 = _force_at_time(time_s, force, t0 + 0.10)
    f_200 = _force_at_time(time_s, force, t0 + 0.20)

    rfd_0_50 = float((f_50 - f0) / 0.05)
    rfd_0_100 = float((f_100 - f0) / 0.10)
    rfd_0_200 = float((f_200 - f0) / 0.20)

    # -----------------------
    # 3) Impulse 0–200ms
    # -----------------------
    mask_200 = time_s <= (t0 + 0.20)
    t_window = time_s[mask_200]
    f_window = force[mask_200]

    # Total impulse (includes BW)
    impulse_0_200_ns = _impulse_trapz(t_window, f_window)

    # Net impulse (above BW)
    net_force_window = f_window - bw_n
    net_impulse_0_200_ns = _impulse_trapz(t_window, net_force_window)

    # -----------------------
    # 4) Normalized metrics
    # -----------------------
    peak_force_xbw = float(peak_force / bw_n) if bw_n != 0 else float("nan")
    net_peak_force_n = float(peak_force - bw_n)

    return {
        "bw_n": bw_n,
        "peak_force_n": peak_force,
        "peak_force_xbw": peak_force_xbw,
        "net_peak_force_n": net_peak_force_n,
        "time_to_peak_ms": time_to_peak_ms,
        "rfd_0_50_n_per_s": rfd_0_50,
        "rfd_0_100_n_per_s": rfd_0_100,
        "rfd_0_200_n_per_s": rfd_0_200,
        "impulse_0_200_ns": impulse_0_200_ns,
        "net_impulse_0_200_ns": net_impulse_0_200_ns,
    }


