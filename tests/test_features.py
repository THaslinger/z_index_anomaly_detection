import pandas as pd
from src.features import prepare_iat


def test_prepare_iat_basic():
    df = pd.DataFrame(
        {
            "can_id": [1, 1, 1],
            "_time": pd.to_datetime(
                ["2024-01-01 00:00:00",
                 "2024-01-01 00:00:01",
                 "2024-01-01 00:00:03"]
            ),
        }
    )

    out = prepare_iat(df, window_s=1.0)

    # First event per CAN-ID must be dropped
    assert len(out) == 2

    # IATs should be correct
    assert out["iat"].tolist() == [1.0, 2.0]

    # Window index should be non-negative integers
    assert out["window_idx"].min() >= 0

