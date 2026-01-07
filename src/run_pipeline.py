import logging

from src.detect import z_anomaly, post_aggregate
from src.features import prepare_iat, window_features, filter_min_windows, classify_frequency, robust_stats
from src.io_handler import load_public_csv
from src.metrics import metrics

logger = logging.getLogger(__name__)

def run_pipeline(cfg) -> None:
    """
    Execute the full anomaly detection pipeline on CAN bus data.

    The pipeline performs:
      1) Data loading
      2) Feature engineering (IAT, windowing, robust statistics)
      3) Z-index based anomaly detection
      4) Temporal and cross-ID aggregation
      5) Metric computation against window-level ground truth

    Parameters
    ----------
    cfg :
        Configuration object providing `data` and `pipeline` settings.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no data remains after filtering or required columns are missing.
    """

    logger.info("Starting pipeline")

    # 1. Step load Data
    df = load_public_csv(cfg.data.path)

    required_cols = {"_time", "can_id", "label_raw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    # 2. Step Feature Engineering
    # 2a Calculating IAT
    df_p = prepare_iat(df, window_s=cfg.pipeline.WINDOW_S)

    # --------------------
    # 2b Filtering Windows

    df_p = filter_min_windows(df_p, min_windows=cfg.pipeline.MIN_WINDOWS)

    if df_p.empty:
        raise ValueError("No data left after min_windows filtering")

    # ------------------------------------------------
    # 2c Calculating statistics and Classify frequency

    stats = robust_stats(
        df_p,
        mad_eps=cfg.pipeline.MAD_EPS,
        z_thresh=cfg.pipeline.Z_THRESH,
        iterations=cfg.pipeline.BASE_ITER,
    )

    stats = classify_frequency(stats, low_freq_s=cfg.pipeline.LOW_FREQ_S)

    # -------------------------------------------
    # 2d Calculating IAT-Median and Message count
    win_feat = window_features(df_p)

    logger.debug(
        "Window features: windows=%d, msg_cnt[min/med/max]=(%d / %.1f / %d)",
        len(win_feat),
        win_feat["msg_cnt"].min(),
        win_feat["msg_cnt"].median(),
        win_feat["msg_cnt"].max(),
    )

    # ---------------------------------------
    # 3. Step z-index based anomaly detection

    res = z_anomaly(
        win_feat,
        stats,
        min_msg=cfg.pipeline.MIN_MSG,
        z_thresh=cfg.pipeline.Z_THRESH,
        mad_eps=cfg.pipeline.MAD_EPS,
    )

    # --------------------------------------------
    # 4. Step Post-aggregation (temporal + voting)

    res = post_aggregate(res, consec_min=cfg.pipeline.CONSEC_MIN, id_vote=cfg.pipeline.ID_VOTE_PER_SEC)

    # -------------------------------------
    # 5. Calculating and Displaying Metrics
    if "label_raw" not in df_p.columns:
        logger.warning("No ground truth labels found (label_raw missing)")
    else:
        gt_by_window = (
            df_p.assign(gt_attack=df_p["label_raw"].astype(str).ne("R"))
            .groupby("window_idx")["gt_attack"]
            .any()
        )

        report = metrics(res, gt_by_window)
        logger.info(
            "Metrics: TP=%d FP=%d FN=%d TN=%d | P=%.3f R=%.3f F1=%.3f",
            report["tp"],
            report["fp"],
            report["fn"],
            report["tn"],
            report["precision"],
            report["recall"],
            report["f1"],
        )

    logger.info("Pipeline finished successfully")

    return
