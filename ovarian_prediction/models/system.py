import os
from typing import Dict

import pandas as pd

from ovarian_prediction.config import MODEL_NAMES

from .xgboost_model import XGBSubmodel


class OvarianMLSystem:
    """Coordinator for the four XGBoost submodels."""

    def __init__(self, n_trials: int = 50, random_state: int = 777):
        self.n_trials = n_trials
        self.random_state = random_state
        self.models: Dict[str, XGBSubmodel] = {
            "PORDM": XGBSubmodel("PORDM", "POR", n_trials, random_state=random_state),
            "HORDM": XGBSubmodel("HORDM", "HOR", n_trials, random_state=random_state),
            "PORSM": XGBSubmodel("PORSM", "POR", n_trials, random_state=random_state),
            "HORSM": XGBSubmodel("HORSM", "HOR", n_trials, random_state=random_state),
        }

    def train_all(self, data: Dict[str, pd.DataFrame], tune: bool = True) -> "OvarianMLSystem":
        mapping = {
            "PORDM": "pordm_train",
            "HORDM": "hordm_train",
            "PORSM": "porsm_train",
            "HORSM": "horsm_train",
        }
        for name, data_key in mapping.items():
            print(f"\n{'=' * 50}")
            print(f"训练子模型: {name}")
            train_df = data[data_key]
            if tune:
                self.models[name].tune_and_fit(train_df)
            else:
                self.models[name].fit(
                    train_df,
                    params={
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                )
        return self

    def evaluate_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        mapping = {
            "PORDM": "pordm_test",
            "HORDM": "hordm_test",
            "PORSM": "porsm_test",
            "HORSM": "horsm_test",
        }
        return {name: self.models[name].evaluate(data[data_key]) for name, data_key in mapping.items()}

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for model in self.models.values():
            model.save(directory)
        print(f"✓ 全部模型已保存至: {directory}")

    @classmethod
    def load(cls, directory: str) -> "OvarianMLSystem":
        system = cls.__new__(cls)
        system.n_trials = 50
        system.random_state = 777
        system.models = {}
        for name, target in zip(MODEL_NAMES, ("POR", "HOR", "POR", "HOR")):
            paths = XGBSubmodel.artifact_paths(directory, name)
            if paths["model"].exists() and paths["metadata"].exists():
                system.models[name] = XGBSubmodel.load_native(paths["model"], paths["metadata"])
                continue

            if not paths["legacy"].exists():
                raise FileNotFoundError(f"缺少模型文件: {paths['model']} 或 {paths['legacy']}")

            legacy_model = XGBSubmodel.load(str(paths["legacy"]))
            if legacy_model.model_ is not None:
                legacy_model.save(directory)
            system.models[name] = legacy_model
        return system
