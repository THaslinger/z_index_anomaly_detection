import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_public_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a public CAN bus CSV file and normalize its timestamp column.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the loaded data with a cleaned '_time' column.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the '_time' column is missing.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading CSV data from %s", path)
    df = pd.read_csv(path)

    if "_time" not in df.columns:
        raise ValueError("load_public_csv: missing required column '_time'")

    n_before = len(df)

    df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["_time"])

    n_after = len(df)
    if n_after < n_before:
        logger.debug(
            "load_public_csv: dropped %d rows with invalid timestamps",
            n_before - n_after,
        )

    # Convert to timezone-naive timestamps for simpler downstream handling
    df["_time"] = df["_time"].dt.tz_localize(None)

    logger.info("Loaded CSV: rows=%d columns=%d", len(df), df.shape[1])
    return df
