import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .metrics import brier_score, split_xy

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


class XGBSubmodel:
    """One XGBoost binary classifier used inside the four-model system."""

    MODEL_FILE_SUFFIX = ".json"
    METADATA_FILE_SUFFIX = ".meta.json"
    LEGACY_FILE_SUFFIX = ".pkl"

    def __init__(
        self,
        name: str,
        target: str,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 777,
    ):
        self.name = name
        self.target = target
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params_: Optional[Dict] = None
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: Optional[list] = None

    @classmethod
    def artifact_paths(cls, directory: str, name: str):
        base_dir = Path(directory)
        return {
            "model": base_dir / f"{name}{cls.MODEL_FILE_SUFFIX}",
            "metadata": base_dir / f"{name}{cls.METADATA_FILE_SUFFIX}",
            "legacy": base_dir / f"{name}{cls.LEGACY_FILE_SUFFIX}",
        }

    def _build_metadata(self) -> Dict:
        return {
            "name": self.name,
            "target": self.target,
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "random_state": self.random_state,
            "best_params": self.best_params_,
            "feature_names": self.feature_names_,
            "model_params": self.model_.get_params() if self.model_ is not None else None,
        }

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 40),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.1, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "eval_metric": "auc",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        clf = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    def tune(self, train_df: pd.DataFrame) -> "XGBSubmodel":
        X, y = split_xy(train_df, self.target)
        self.feature_names_ = list(X.columns)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )
        self.best_params_ = study.best_params
        print(f"[{self.name}] 最佳 AUC (CV): {study.best_value:.4f}")
        print(f"[{self.name}] 最佳参数: {self.best_params_}")
        return self

    def fit(self, train_df: pd.DataFrame, params: Optional[Dict] = None) -> "XGBSubmodel":
        X, y = split_xy(train_df, self.target)
        self.feature_names_ = list(X.columns)

        final_params = {
            "eval_metric": "auc",
            "random_state": self.random_state,
            "n_jobs": -1,
            **(params or self.best_params_ or {}),
        }
        self.model_ = xgb.XGBClassifier(**final_params)
        self.model_.fit(X, y)
        return self

    def tune_and_fit(self, train_df: pd.DataFrame) -> "XGBSubmodel":
        self.tune(train_df)
        self.fit(train_df)
        return self

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            return df.astype(float)
        X = df.reindex(columns=self.feature_names_, fill_value=0)
        return X.astype(float)

    def predict_proba(self, df: pd.DataFrame):
        if self.model_ is None:
            raise RuntimeError(f"{self.name} 尚未训练")
        X = self._align_features(df)
        return self.model_.predict_proba(X)[:, 1]

    def predict_class(self, df: pd.DataFrame, threshold: float = 0.5):
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        if self.model_ is None:
            raise RuntimeError(f"{self.name} 尚未训练")
        X_test, y_test = split_xy(test_df, self.target)
        proba = self.model_.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        bs = brier_score(y_test, proba)
        print(f"[{self.name}] 测试集 AUC: {auc:.4f} | Brier Score: {bs:.4f}")
        return {"auc": auc, "brier_score": bs}

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if self.model_ is None:
            raise RuntimeError(f"{self.name} 尚未训练，无法保存")

        paths = self.artifact_paths(directory, self.name)
        self.model_.save_model(paths["model"])
        with paths["metadata"].open("w", encoding="utf-8") as fp:
            json.dump(self._build_metadata(), fp, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "XGBSubmodel":
        artifact_path = Path(path)
        if artifact_path.suffix == cls.LEGACY_FILE_SUFFIX:
            return cls._load_legacy_pickle(artifact_path)
        if artifact_path.suffix == cls.MODEL_FILE_SUFFIX:
            metadata_path = artifact_path.with_name(f"{artifact_path.stem}{cls.METADATA_FILE_SUFFIX}")
            return cls.load_native(artifact_path, metadata_path)
        raise ValueError(f"不支持的模型文件格式: {artifact_path}")

    @classmethod
    def load_native(cls, model_path: Path, metadata_path: Path) -> "XGBSubmodel":
        if not model_path.exists():
            raise FileNotFoundError(f"缺少模型文件: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"缺少模型元数据文件: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = cls(
            name=metadata["name"],
            target=metadata["target"],
            n_trials=metadata.get("n_trials", 50),
            cv_folds=metadata.get("cv_folds", 5),
            random_state=metadata.get("random_state", 777),
        )
        model.best_params_ = metadata.get("best_params")
        model.feature_names_ = metadata.get("feature_names")

        clf = xgb.XGBClassifier(**(metadata.get("model_params") or {}))
        clf.load_model(model_path)
        model.model_ = clf
        return model

    @classmethod
    def _load_legacy_pickle(cls, path: Path) -> "XGBSubmodel":
        # Legacy .pkl files may contain serialized XGBoost internals that emit
        # compatibility warnings on newer versions. Silence them during the
        # one-time migration path and prefer native model files afterwards.
        previous_config = xgb.get_config()
        try:
            xgb.set_config(verbosity=0)
            model = joblib.load(path)
        finally:
            xgb.set_config(**previous_config)

        if not isinstance(model, cls):
            raise TypeError(f"无法从 {path} 加载 XGBSubmodel")
        if not hasattr(model, "feature_names_"):
            model.feature_names_ = None
        if not hasattr(model, "best_params_"):
            model.best_params_ = None
        return model
