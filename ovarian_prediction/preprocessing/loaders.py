import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Read the raw Excel data and normalize object columns to categoricals."""
    df = pd.read_excel(filepath)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df
