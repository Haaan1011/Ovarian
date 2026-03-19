from typing import Optional

import pandas as pd

from .feature_sets import CATEGORICAL_COLS


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: Optional[list] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """One-hot encode categorical columns using pandas dummies."""
    cols = cat_cols or [c for c in CATEGORICAL_COLS if c in df.columns]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return pd.get_dummies(
        df,
        columns=present,
        drop_first=drop_first,
        dtype=float,
    )
