import pandas as pd
from src.metrics import metrics


def test_metrics_basic_counts():
    res = pd.DataFrame(
        {
            "window_idx": [0, 1, 2, 3],
            "alarm_final": [True, False, True, False],
        }
    )

    gt_by_window = pd.Series(
        [True, False, False, True],
        index=[0, 1, 2, 3],
    )

    report = metrics(res, gt_by_window)

    assert report["tp"] == 1
    assert report["fp"] == 1
    assert report["fn"] == 1
    assert report["tn"] == 1

    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["f1"] == 0.5
