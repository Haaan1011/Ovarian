from typing import Optional

import pandas as pd

from .feature_sets import COLS_TO_DROP


def drop_unused_columns(
    df: pd.DataFrame,
    extra_drop: Optional[list] = None,
) -> pd.DataFrame:
    """Drop columns that were excluded in the original R workflow."""
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if extra_drop:
        to_drop += [c for c in extra_drop if c in df.columns]
    return df.drop(columns=to_drop)
