import pandas as pd
from src.detect import z_anomaly


def test_z_anomaly_cyclic_triggers():
    win_df = pd.DataFrame(
        {
            "can_id": [1, 1],
            "window_idx": [0, 1],
            "iat_med": [1.0, 10.0],   # second window is extreme
            "msg_cnt": [5, 5],
        }
    )

    stats = pd.DataFrame(
        {
            "can_id": [1],
            "mu": [1.0],
            "sigma": [0.1],
            "freq_class": ["cyclic"],
        }
    )

    res = z_anomaly(
        win_df,
        stats,
        min_msg=1,
        z_thresh=3.0,
        mad_eps=1e-4,
    )

    assert res["anomaly_z"].tolist() == [False, True]
