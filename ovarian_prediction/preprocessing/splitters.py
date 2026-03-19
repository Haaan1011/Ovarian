from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.3,
    random_state: int = 777,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset with stratification on the target label."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
