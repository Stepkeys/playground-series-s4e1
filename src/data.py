from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_competition_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the competition training and test datasets."""
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "test.csv")
    return train, test