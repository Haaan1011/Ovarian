"""
models.py
=========
模型训练模块, 对应 R 中的 ML system (part 1).Rmd

功能:
  - 4 个 XGBoost 子模型封装 (PORDM, HORDM, PORSM, HORSM)
  - Optuna 贝叶斯超参数调优 (对应 R 的 tune_bayes)
  - 5-fold 分层交叉验证 (对应 vfold_cv)
  - AUC + Brier Score 评估
  - 模型持久化 (joblib)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import optuna
import joblib
import os
from typing import Dict, Tuple, Optional
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """将 DataFrame 拆分为特征矩阵 X 和二值标签 y."""
    X = df.drop(columns=[target])
    # 直接映射: Yes -> 1, No -> 0, 避免LabelEncoder顺序不确定
    y_raw = df[target].astype(str).str.strip()
    y = np.where(y_raw == "Yes", 1, 0).astype(int)
    return X.astype(float), y


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier 评分 = mean((obs - pred)^2). 对应 R: brier_score()"""
    return float(np.mean((y_true - y_prob) ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# 单个 XGBoost 子模型
# ──────────────────────────────────────────────────────────────────────────────

class XGBSubmodel:
    """
    封装单个 XGBoost 二分类子模型.
    支持贝叶斯超参数调优 (Optuna) 和 5-fold CV.
    """

    def __init__(self,
                 name: str,
                 target: str,
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 777):
        self.name = name
        self.target = target
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params_: Optional[Dict] = None
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: Optional[list] = None

    # ── 贝叶斯调参 ─────────────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial,
                   X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Optuna 目标函数. 搜索空间对应 R 的 tune_bayes 超参范围.
        """
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":         trial.suggest_int("max_depth", 1, 15),
            "min_child_weight":  trial.suggest_int("min_child_weight", 2, 40),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-5, 0.1, log=True),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "subsample":         trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",
                                                      0.1, 1.0),
            "eval_metric":       "auc",
            "random_state":      self.random_state,
            "n_jobs":            -1,
        }
        clf = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scores = cross_val_score(clf, X, y, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    def tune(self, train_df: pd.DataFrame) -> "XGBSubmodel":
        """
        对训练集运行 Optuna 贝叶斯调参.
        对应 R: tune_bayes(..., iter=100, metrics=roc_auc, no_improve=50)
        """
        X, y = _split_xy(train_df, self.target)
        self.feature_names_ = list(X.columns)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=False
        )
        self.best_params_ = study.best_params
        print(f"[{self.name}] 最佳 AUC (CV): {study.best_value:.4f}")
        print(f"[{self.name}] 最佳参数: {self.best_params_}")
        return self

    # ── 训练最终模型 ────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame,
            params: Optional[Dict] = None) -> "XGBSubmodel":
        """
        在整个训练集上拟合最终模型.
        对应 R: finalize_workflow(...) %>% fit(train_data)
        """
        X, y = _split_xy(train_df, self.target)
        self.feature_names_ = list(X.columns)

        p = params or self.best_params_ or {}
        final_params = {
            "eval_metric": "auc",
            "random_state": self.random_state,
            "n_jobs": -1,
            **p
        }
        self.model_ = xgb.XGBClassifier(**final_params)
        self.model_.fit(X, y)
        return self

    def tune_and_fit(self, train_df: pd.DataFrame) -> "XGBSubmodel":
        """调参 + 拟合一步完成."""
        self.tune(train_df)
        self.fit(train_df)
        return self

    # ── 预测 ───────────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        返回 POR/HOR 为 'Yes' 的概率.
        对应 R: predict(..., type='prob')[, '.pred_Yes']
        """
        X = self._align_features(df)
        return self.model_.predict_proba(X)[:, 1]

    def predict_class(self, df: pd.DataFrame,
                      threshold: float = 0.5) -> np.ndarray:
        """返回二值预测类别."""
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保预测时特征列与训练时一致 (XGBoost 支持 NaN)."""
        if self.feature_names_ is None:
            return df.astype(float)
        X = df.reindex(columns=self.feature_names_, fill_value=0)
        return X.astype(float)

    # ── 评估 ───────────────────────────────────────────────────────────────────

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算测试集 AUC 和 Brier score.
        对应 R: pROC::roc(...) 和 brier_score()
        """
        X_test, y_test = _split_xy(test_df, self.target)
        proba = self.model_.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        bs = brier_score(y_test, proba)
        print(f"[{self.name}] 测试集 AUC: {auc:.4f} | Brier Score: {bs:.4f}")
        return {"auc": auc, "brier_score": bs}

    # ── 持久化 ─────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, f"{self.name}.pkl"))

    @classmethod
    def load(cls, path: str) -> "XGBSubmodel":
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# 4 个子模型管理器
# ──────────────────────────────────────────────────────────────────────────────

class OvarianMLSystem:
    """
    集成 4 个 XGBoost 子模型的管理器:
      - PORDM: 卵巢低反应诊断模型 (无干预特征)
      - HORDM: 卵巢高反应诊断模型 (无干预特征)
      - PORSM: 卵巢低反应策略模型 (含干预特征)
      - HORSM: 卵巢高反应策略模型 (含干预特征)

    对应 R: pordm_top_final, hordm_top_final, porsm_top_final, horsm_top_final
    """

    def __init__(self, n_trials: int = 50, random_state: int = 777):
        self.n_trials = n_trials
        self.random_state = random_state
        self.models: Dict[str, XGBSubmodel] = {
            "PORDM": XGBSubmodel("PORDM", "POR", n_trials, random_state=random_state),
            "HORDM": XGBSubmodel("HORDM", "HOR", n_trials, random_state=random_state),
            "PORSM": XGBSubmodel("PORSM", "POR", n_trials, random_state=random_state),
            "HORSM": XGBSubmodel("HORSM", "HOR", n_trials, random_state=random_state),
        }

    def train_all(self, data: Dict[str, pd.DataFrame],
                  tune: bool = True) -> "OvarianMLSystem":
        """
        训练全部 4 个子模型.

        Args:
            data: OvarianPreprocessor.fit_transform() 返回的字典
            tune: True=先调参再训练; False=用默认参数直接训练(快速测试)
        """
        mapping = {
            "PORDM": "pordm_train",
            "HORDM": "hordm_train",
            "PORSM": "porsm_train",
            "HORSM": "horsm_train",
        }
        for name, data_key in mapping.items():
            print(f"\n{'='*50}")
            print(f"训练子模型: {name}")
            train_df = data[data_key]
            if tune:
                self.models[name].tune_and_fit(train_df)
            else:
                # 快速训练 (默认参数, 不调参)
                self.models[name].fit(train_df, params={
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                })
        return self

    def evaluate_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """评估全部 4 个子模型."""
        mapping = {
            "PORDM": "pordm_test",
            "HORDM": "hordm_test",
            "PORSM": "porsm_test",
            "HORSM": "horsm_test",
        }
        results = {}
        for name, data_key in mapping.items():
            results[name] = self.models[name].evaluate(data[data_key])
        return results

    def save(self, directory: str) -> None:
        """保存全部子模型."""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            model.save(directory)
        print(f"✓ 全部模型已保存至: {directory}")

    @classmethod
    def load(cls, directory: str) -> "OvarianMLSystem":
        """从目录读取全部子模型."""
        system = cls.__new__(cls)
        system.models = {}
        for name, target in [("PORDM", "POR"), ("HORDM", "HOR"),
                              ("PORSM", "POR"), ("HORSM", "HOR")]:
            path = os.path.join(directory, f"{name}.pkl")
            system.models[name] = XGBSubmodel.load(path)
        return system
