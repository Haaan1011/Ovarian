import os
from typing import Dict

import pandas as pd

from .cleaners import drop_unused_columns
from .encoders import encode_categoricals
from .feature_sets import HORRM_FEATURES, OS_INTERVENTIONS, PORRM_FEATURES, TARGET_HOR, TARGET_POR
from .imputers import MICEImputer
from .splitters import stratified_split


class OvarianPreprocessor:
    """End-to-end preprocessing pipeline used by the training workflow."""

    def __init__(self, random_state: int = 777):
        self.random_state = random_state
        self.porsm_imputer = MICEImputer(random_state=random_state)
        self.horsm_imputer = MICEImputer(random_state=random_state)

    def _select_porsm(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = PORRM_FEATURES + OS_INTERVENTIONS + [TARGET_POR]
        return df[[c for c in cols if c in df.columns]].copy()

    def _select_horsm(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = HORRM_FEATURES + OS_INTERVENTIONS + [TARGET_HOR]
        return df[[c for c in cols if c in df.columns]].copy()

    def fit_transform(self, raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df = drop_unused_columns(raw_df)

        porsm_df = self._select_porsm(df)
        porsm_df[TARGET_POR] = porsm_df[TARGET_POR].astype(str)
        porsm_train_raw, porsm_test_raw = stratified_split(
            porsm_df,
            TARGET_POR,
            random_state=self.random_state,
        )

        porsm_train = self.porsm_imputer.fit_transform(porsm_train_raw)
        porsm_test = self.porsm_imputer.transform(porsm_test_raw)
        porsm_train = encode_categoricals(porsm_train)
        porsm_test = encode_categoricals(porsm_test)

        horsm_df = self._select_horsm(df)
        horsm_df[TARGET_HOR] = horsm_df[TARGET_HOR].astype(str)
        horsm_train_raw, horsm_test_raw = stratified_split(
            horsm_df,
            TARGET_HOR,
            random_state=self.random_state,
        )

        horsm_train = self.horsm_imputer.fit_transform(horsm_train_raw)
        horsm_test = self.horsm_imputer.transform(horsm_test_raw)
        horsm_train = encode_categoricals(horsm_train)
        horsm_test = encode_categoricals(horsm_test)

        os_cols_dummy = [
            c for c in porsm_train.columns if any(c.startswith(p) for p in OS_INTERVENTIONS)
        ]
        pordm_train = porsm_train.drop(columns=os_cols_dummy, errors="ignore")
        pordm_test = porsm_test.drop(columns=os_cols_dummy, errors="ignore")

        os_cols_dummy_h = [
            c for c in horsm_train.columns if any(c.startswith(p) for p in OS_INTERVENTIONS)
        ]
        hordm_train = horsm_train.drop(columns=os_cols_dummy_h, errors="ignore")
        hordm_test = horsm_test.drop(columns=os_cols_dummy_h, errors="ignore")

        return {
            "porsm_train": porsm_train,
            "porsm_test": porsm_test,
            "horsm_train": horsm_train,
            "horsm_test": horsm_test,
            "pordm_train": pordm_train,
            "pordm_test": pordm_test,
            "hordm_train": hordm_train,
            "hordm_test": hordm_test,
        }

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.porsm_imputer.save(os.path.join(directory, "porsm_imputer.pkl"))
        self.horsm_imputer.save(os.path.join(directory, "horsm_imputer.pkl"))

    @classmethod
    def load(cls, directory: str) -> "OvarianPreprocessor":
        proc = cls.__new__(cls)
        proc.porsm_imputer = MICEImputer.load(os.path.join(directory, "porsm_imputer.pkl"))
        proc.horsm_imputer = MICEImputer.load(os.path.join(directory, "horsm_imputer.pkl"))
        return proc
