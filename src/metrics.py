import logging
import pandas as pd

logger = logging.getLogger(__name__)

def metrics(
    res: pd.DataFrame,
    gt_by_window: pd.Series,
    *,
    win_col: str = "window_idx",
    pred_col: str = "alarm_final",
) -> dict[str, float | int]:
    """
    Compute binary classification metrics at window level.

    Compares predicted anomaly alarms against ground-truth attack labels aggregated per time window.

    Parameters
    ----------
    res:
        Result dataframe containing window-level predictions.
        Must include columns [win_col, pred_col].
    gt_by_window:
        Boolean Series indexed by window index, indicating whether a window contains at least one ground-truth attack.
    win_col:
        Column in `res` identifying the window index.
    pred_col:
        Column in `res` containing the predicted alarm (boolean-like).

    Returns
    -------
    dict
        Dictionary with:
        - tp, tn, fp, fn
        - precision, recall, f1

    Raises
    ------
    ValueError
        If required columns are missing or inputs are inconsistent.
    """
    required_cols = {win_col, pred_col}
    missing = required_cols - set(res.columns)
    if missing:
        raise ValueError(f"res missing required columns: {sorted(missing)}")
    if gt_by_window.empty:
        logger.warning("metrics: gt_by_window is empty; all windows treated as non-attack")

    scored = res.merge(
        gt_by_window.rename("gt_attack"),
        left_on=win_col,
        right_index=True,
        how="left",
    ).fillna({"gt_attack": False})

    tp = int((scored[pred_col] & scored["gt_attack"]).sum())
    tn = int((~scored[pred_col] & ~scored["gt_attack"]).sum())
    fp = int((scored[pred_col] & ~scored["gt_attack"]).sum())
    fn = int((~scored[pred_col] & scored["gt_attack"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
