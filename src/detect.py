import logging
import pandas as pd

logger = logging.getLogger(__name__)

def z_anomaly(
        win_df: pd.DataFrame,
        stats: pd.DataFrame,
        *,
        min_msg: int,
        z_thresh: float,
        mad_eps: float
) -> pd.DataFrame:
    """
    Compute per-window anomaly flags using a robust z-score model.

    For CAN IDs classified as "cyclic", a robust z-score is computed from the window median IAT (iat_med)
    and robust parameters (mu, sigma).
    For "low_freq" CAN IDs, windows are flagged as anomalous if msg_cnt > 1.

    Parameters
    ----------
    win_df:
        Window-level features with at least: [can_id, window_idx, iat_med, msg_cnt].
    stats:
        Per-CAN-ID robust statistics with at least: [can_id, mu, sigma, freq_class].
    min_msg:
        Minimum number of messages per window to consider for scoring.
    z_thresh:
        Absolute z-score threshold for anomalies (cyclic IDs only).
    mad_eps:
        Small epsilon to replace sigma==0 to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        Input windows merged with stats and additional columns:
        [z, anomaly_z] (z only for cyclic windows).
    """
    required_win = {"can_id", "iat_med", "msg_cnt"}
    required_stats = {"can_id", "mu", "sigma", "freq_class"}

    missing_win = required_win - set(win_df.columns)
    missing_stats = required_stats - set(stats.columns)
    if missing_win:
        raise ValueError(f"win_df missing required columns: {sorted(missing_win)}")
    if missing_stats:
        raise ValueError(f"stats missing required columns: {sorted(missing_stats)}")
    if min_msg < 1:
        raise ValueError("min_msg must be >= 1")
    if z_thresh <= 0:
        raise ValueError("z_thresh must be > 0")
    if mad_eps <= 0:
        raise ValueError("mad_eps must be > 0")

    logger.debug(
        "z_anomaly: start windows=%d, can_ids=%d | min_msg=%d z_thresh=%.3f mad_eps=%.1e",
        len(win_df),
        win_df["can_id"].nunique(dropna=True),
        min_msg,
        z_thresh,
        mad_eps,
    )

    merged = win_df.merge(stats, on="can_id", how="left")

    # If stats are missing for some IDs, scoring will be incomplete.
    missing_stats_rows = merged["freq_class"].isna().sum()
    if missing_stats_rows:
        logger.warning(
            "z_anomaly: %d windows have no matching stats (missing can_id in stats)",
            int(missing_stats_rows),
        )

    # Avoid division by zero for pathological sigma.
    merged["sigma"] = merged["sigma"].replace(0, mad_eps)

    before = len(merged)
    merged = merged[merged["msg_cnt"] >= min_msg].copy()
    logger.debug("z_anomaly: filtered by min_msg: kept=%d dropped=%d", len(merged), before - len(merged))

    # Cyclic scoring: robust z-score on IAT median.
    cyc_mask = merged["freq_class"] == "cyclic"
    n_cyc = int(cyc_mask.sum())
    if n_cyc:
        merged.loc[cyc_mask, "z"] = (
            (merged.loc[cyc_mask, "iat_med"] - merged.loc[cyc_mask, "mu"])
            / merged.loc[cyc_mask, "sigma"]
        )
        merged.loc[cyc_mask, "anomaly_z"] = merged.loc[cyc_mask, "z"].abs() > z_thresh
    else:
        logger.debug("z_anomaly: no cyclic windows to score")

    # Low-frequency heuristic.
    low_mask = merged["freq_class"] == "low_freq"
    n_low = int(low_mask.sum())
    if n_low:
        merged.loc[low_mask, "anomaly_z"] = merged.loc[low_mask, "msg_cnt"] > 1
    else:
        logger.debug("z_anomaly: no low_freq windows to score")

    merged["anomaly_z"] = merged["anomaly_z"].astype("boolean").fillna(False)

    n_anom = int(merged["anomaly_z"].sum())
    logger.info(
        "z_anomaly: done windows=%d anomalies=%d (%.2f%%)",
        len(merged),
        n_anom,
        (100.0 * n_anom / len(merged)) if len(merged) else 0.0,
    )

    return merged


def post_aggregate(
    res: pd.DataFrame,
    *,
    consec_min: int,
    id_vote: int,
    win_col: str = "window_idx",
) -> pd.DataFrame:
    """
    Aggregate per-window anomaly flags into more stable alarms.

    Two alarm channels are computed:
      - alarm_consec: triggers if an ID has >= consec_min anomalies in a rolling window.
      - alarm_vote: triggers if a window has >= id_vote anomalous CAN IDs.

    The final alarm is the OR of both channels.

    Parameters
    ----------
    res:
        Result dataframe with at least: [can_id, win_col, anomaly_z].
    consec_min:
        Minimum number of anomalous windows in the rolling history to trigger alarm_consec.
    id_vote:
        Minimum number of anomalous CAN IDs in a window to trigger alarm_vote.
    win_col:
        Column name for the window index (default: "window_idx").

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with: [consec, alarm_consec, ids_in_win, alarm_vote, alarm_final].
    """
    required = {"can_id", win_col, "anomaly_z"}
    missing = required - set(res.columns)
    if missing:
        raise ValueError(f"res missing required columns: {sorted(missing)}")
    if consec_min < 1:
        raise ValueError("consec_min must be >= 1")
    if id_vote < 1:
        raise ValueError("id_vote must be >= 1")

    logger.debug(
        "post_aggregate: start rows=%d can_ids=%d consec_min=%d id_vote=%d",
        len(res),
        res["can_id"].nunique(dropna=True),
        consec_min,
        id_vote,
    )

    out = res.sort_values(["can_id", win_col]).copy()

    # Rolling count of anomalies per CAN ID (consecutive-ish signal).
    out["consec"] = out.groupby("can_id")["anomaly_z"].transform(
        lambda s: s.rolling(consec_min, min_periods=1).sum()
    )
    out["alarm_consec"] = out["consec"] >= consec_min

    # Voting signal: how many IDs are anomalous in the same window.
    out["ids_in_win"] = out.groupby(win_col)["anomaly_z"].transform("sum")
    out["alarm_vote"] = out["ids_in_win"] >= id_vote

    out["alarm_final"] = out["alarm_consec"] | out["alarm_vote"]

    n_final = int(out["alarm_final"].sum())
    logger.info(
        "post_aggregate: done rows=%d alarm_final=%d (%.2f%%)",
        len(out),
        n_final,
        (100.0 * n_final / len(out)) if len(out) else 0.0,
    )

    return out