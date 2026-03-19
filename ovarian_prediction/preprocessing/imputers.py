from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class MICEImputer:
    """Random-forest based iterative imputer for the clinical tabular data."""

    def __init__(self, max_iter: int = 10, random_state: int = 777):
        self.max_iter = max_iter
        self.random_state = random_state
        self._num_imputer: Optional[IterativeImputer] = None
        self._cat_cols: list = []
        self._num_cols: list = []
        self._cat_modes: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "MICEImputer":
        self._cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
        self._num_cols = df.select_dtypes(include="number").columns.tolist()

        for col in self._cat_cols:
            self._cat_modes[col] = df[col].mode()[0]

        if self._num_cols:
            self._num_imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self._num_imputer.fit(df[self._num_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self._cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self._cat_modes.get(col, df[col].mode()[0]))

        if self._num_imputer and self._num_cols:
            available_num = [c for c in self._num_cols if c in df.columns]
            imputed = self._num_imputer.transform(df[available_num])
            df[available_num] = imputed
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MICEImputer":
        return joblib.load(path)
