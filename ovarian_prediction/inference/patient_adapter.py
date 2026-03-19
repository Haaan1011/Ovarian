from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ovarian_prediction.preprocessing import (
    CATEGORICAL_COLS,
    encode_categoricals,
)


def patient_dict_to_df(
    patient: Dict[str, Any],
    features: list,
    cat_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Convert a patient dict into an encoded single-row dataframe."""
    row = {feature: patient.get(feature, np.nan) for feature in features}
    row = {key: (np.nan if value is None else value) for key, value in row.items()}
    df = pd.DataFrame([row])

    cats = cat_cols or [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in cats:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val == "nan":
                df[col] = "Unknown"
            df[col] = df[col].astype(str).astype("category")

    return encode_categoricals(df, cat_cols=cats, drop_first=True)
