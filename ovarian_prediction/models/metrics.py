from typing import Tuple

import numpy as np
import pandas as pd


def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split a dataframe into features and binary labels."""
    X = df.drop(columns=[target])
    y_raw = df[target].astype(str).str.strip()
    y = np.where(y_raw == "Yes", 1, 0).astype(int)
    return X.astype(float), y


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))
