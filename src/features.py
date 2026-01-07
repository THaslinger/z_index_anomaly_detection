import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_iat(
    df: pd.DataFrame,
    *,
    time_col: str = "_time",
    id_col: str = "can_id",
    window_s: float = 1.0,
) -> pd.DataFrame:
    """
    Prepare inter-arrival-time (IAT) features and assign time windows.

    For each CAN ID, the inter-arrival time (iat) is computed as the difference
    between consecutive timestamps. Each row is then assigned to a fixed-length
    window based on seconds since the first timestamp.

    Parameters
    ----------
    df:
        Raw event-level dataframe containing at least [time_col, id_col].
        `time_col` must be datetime-like.
    time_col:
        Timestamp column name.
    id_col:
        CAN identifier column name.
    window_s:
        Window size in seconds (must be > 0).

    Returns
    -------
    pd.DataFrame
        Copy of `df` sorted by [id_col, time_col] with added columns:
        - iat: inter-arrival time in seconds (first event per id is dropped)
        - sec_from_start: seconds since global start timestamp
        - window_idx: integer window index

    Raises
    ------
    ValueError
        If required columns are missing or parameters are invalid.
    """
    if window_s <= 0:
        raise ValueError("window_s must be > 0")

    required = {time_col, id_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prepare_iat: missing required columns: {sorted(missing)}")

    logger.debug(
        "prepare_iat: start rows=%d ids=%d window_s=%.3f",
        len(df),
        df[id_col].nunique(dropna=True),
        window_s,
    )

    out = df.sort_values([id_col, time_col]).copy()
    out["iat"] = out.groupby(id_col)[time_col].diff().dt.total_seconds()
    before = len(out)
    out = out.dropna(subset=["iat"])

    if out.empty:
        logger.warning("prepare_iat: no rows left after dropping first event per id")
        return out

    t0 = out[time_col].min()
    out["sec_from_start"] = (out[time_col] - t0).dt.total_seconds()
    out["window_idx"] = (out["sec_from_start"] // window_s).astype(int)

    logger.debug("prepare_iat: done rows=%d dropped=%d windows=%d",
                 len(out), before - len(out), out["window_idx"].nunique(dropna=True))
    return out


def filter_min_windows(
    df: pd.DataFrame,
    *,
    can_id: str = "can_id",
    window_col: str = "window_idx",
    min_windows: int = 5,
) -> pd.DataFrame:
    """
    Filter CAN IDs that do not appear in at least `min_windows` distinct windows.

    This is used to remove sparse IDs that would lead to unreliable statistics.

    Parameters
    ----------
    df:
        Prepared dataframe containing [can_id, window_col].
    can_id:
        CAN identifier column name.
    window_col:
        Window index column name.
    min_windows:
        Minimum number of distinct windows required per CAN ID (must be >= 1).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only CAN IDs meeting the requirement.

    Raises
    ------
    ValueError
        If required columns are missing or parameters are invalid.
    """
    if min_windows < 1:
        raise ValueError("min_windows must be >= 1")

    required = {can_id, window_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filter_min_windows: missing required columns: {sorted(missing)}")

    if df.empty:
        logger.warning("filter_min_windows: input is empty")
        return df

    keep_counts = df.groupby(can_id)[window_col].nunique()

    max_windows = int(keep_counts.max()) if not keep_counts.empty else 0
    n_keep = int((keep_counts >= min_windows).sum())

    logger.info("filter_min_windows: keeping %d/%d ids with >= %d windows",
                n_keep, int(keep_counts.size), min_windows)

    keep_ids = keep_counts[keep_counts >= min_windows].index
    out = df[df[can_id].isin(keep_ids)]
    return out


def robust_stats(
    df: pd.DataFrame,
    *,
    id_col: str = "can_id",
    iat_col: str = "iat",
    mad_eps: float = 1e-4,
    z_thresh: float = 3.0,
    iterations: int = 3,
) -> pd.DataFrame:
    """
    Estimate robust per-ID location and scale via iterative MAD-based clipping.

    For each CAN ID, the median (mu) and MAD-based scale (sigma) are computed.
    Outliers are removed based on |z| > z_thresh and the process is repeated.

    Parameters
    ----------
    df:
        Prepared dataframe containing [id_col, iat_col].
    id_col:
        CAN identifier column name.
    iat_col:
        IAT column name (seconds).
    mad_eps:
        Small epsilon used when sigma becomes 0 (must be > 0).
    z_thresh:
        Threshold for clipping by absolute z-score (must be > 0).
    iterations:
        Number of clipping iterations (must be >= 1).

    Returns
    -------
    pd.DataFrame
        Per-ID statistics with columns: [id_col, mu, sigma].

    Raises
    ------
    ValueError
        If required columns are missing or parameters are invalid.
    """
    if mad_eps <= 0:
        raise ValueError("mad_eps must be > 0")
    if z_thresh <= 0:
        raise ValueError("z_thresh must be > 0")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    required = {id_col, iat_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"robust_stats: missing required columns: {sorted(missing)}")

    if df.empty:
        logger.warning("robust_stats: input is empty")
        return pd.DataFrame(columns=[id_col, "mu", "sigma"])

    logger.debug(
        "robust_stats: start rows=%d ids=%d iterations=%d z_thresh=%.2f",
        len(df),
        df[id_col].nunique(dropna=True),
        iterations,
        z_thresh,
    )

    tmp = df[[id_col, iat_col]].copy()
    stats = None

    for i in range(iterations):
        before = len(tmp)

        stats = (
            tmp.groupby(id_col)[iat_col]
            .agg(
                mu="median",
                sigma=lambda x: 1.4826 * (x - x.median()).abs().median(),
            )
            .reset_index()
        )
        stats["sigma"] = stats["sigma"].replace(0, mad_eps)

        tmp = tmp.merge(stats, on=id_col, how="left")
        z = (tmp[iat_col] - tmp["mu"]) / tmp["sigma"]
        tmp = tmp[z.abs() <= z_thresh][[id_col, iat_col]]

        logger.debug("robust_stats: iter=%d kept=%d dropped=%d",
                     i + 1, len(tmp), before - len(tmp))

        if tmp.empty:
            logger.warning("robust_stats: all rows removed during clipping (iter=%d)", i + 1)
            break

    assert stats is not None  # for type-checkers
    logger.info("robust_stats: done ids=%d", len(stats))
    return stats


def classify_frequency(stats: pd.DataFrame, low_freq_s: float) -> pd.DataFrame:
    """
    Classify CAN IDs into frequency regimes based on robust median IAT.

    Parameters
    ----------
    stats:
        Per-ID statistics containing at least column 'mu' (median IAT in seconds).
    low_freq_s:
        Threshold in seconds. IDs with mu > low_freq_s are marked as 'low_freq',
        otherwise 'cyclic'.

    Returns
    -------
    pd.DataFrame
        Copy of `stats` with an additional column 'freq_class'.
    """
    if stats.empty:
        logger.warning("classify_frequency: stats is empty")
        stats = stats.copy()
        stats["freq_class"] = pd.Series(dtype="object")
        return stats

    if "mu" not in stats.columns:
        raise ValueError("classify_frequency: stats missing required column 'mu'")

    out = stats.copy()
    out["freq_class"] = np.where(out["mu"] > low_freq_s, "low_freq", "cyclic")

    logger.debug(
        "classify_frequency: low_freq=%d cyclic=%d (threshold=%.3f)",
        int((out["freq_class"] == "low_freq").sum()),
        int((out["freq_class"] == "cyclic").sum()),
        low_freq_s,
    )
    return out


def window_features(
    df: pd.DataFrame,
    *,
    id_col: str = "can_id",
    win_col: str = "window_idx",
    iat_col: str = "iat",
) -> pd.DataFrame:
    """
    Aggregate event-level IAT data into window-level features per CAN ID.

    Parameters
    ----------
    df:
        Prepared dataframe containing [id_col, win_col, iat_col].
    id_col:
        CAN identifier column name.
    win_col:
        Window index column name.
    iat_col:
        IAT column name.

    Returns
    -------
    pd.DataFrame
        Window-level features with columns:
        - id_col, win_col
        - iat_med: median IAT within the window
        - msg_cnt: number of messages (events) within the window
    """
    required = {id_col, win_col, iat_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"window_features: missing required columns: {sorted(missing)}")

    if df.empty:
        logger.warning("window_features: input is empty")
        return pd.DataFrame(columns=[id_col, win_col, "iat_med", "msg_cnt"])

    out = (
        df.groupby([id_col, win_col])
        .agg(iat_med=(iat_col, "median"), msg_cnt=(iat_col, "size"))
        .reset_index()
    )

    logger.debug("window_features: produced windows=%d", len(out))
    return out
