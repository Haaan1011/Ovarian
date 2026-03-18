"""
predict.py
==========
单患者推理接口, 对应 R 中 predict() 函数的使用方式.

功能:
  - 接受单个患者参数字典
  - 自动对齐特征 (处理缺失值, XGBoost 原生支持 NaN)
  - 调用 PORDM/HORDM 诊断模型 和 PORSM/HORSM 策略模型
  - 返回各子模型的概率预测
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from .preprocessing import (
    PORRM_FEATURES, HORRM_FEATURES, OS_INTERVENTIONS,
    CATEGORICAL_COLS, encode_categoricals
)
from .models import OvarianMLSystem


# ──────────────────────────────────────────────────────────────────────────────
# 特征对齐工具
# ──────────────────────────────────────────────────────────────────────────────

def _patient_dict_to_df(patient: Dict[str, Any],
                        features: list,
                        cat_cols: Optional[list] = None) -> pd.DataFrame:
    """
    将患者参数字典转为已编码的 DataFrame,
    对应 R 中对新数据进行 bake(new_data=...) 的过程.

    缺失字段会填充为 NaN (XGBoost 可原生处理).
    """
    row = {f: patient.get(f, np.nan) for f in features}
    # None -> np.nan
    row = {k: (np.nan if v is None else v) for k, v in row.items()}
    df = pd.DataFrame([row])

    # 分类列编码: 若值为 NaN 则先设为缺失占位再编码
    cats = cat_cols or [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cats:
        if col in df.columns:
            val = df[col].iloc[0]
            if pd.isna(val) or val == "nan":
                df[col] = "Unknown"
            df[col] = df[col].astype(str).astype("category")

    df = encode_categoricals(df, cat_cols=cats, drop_first=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 主预测接口
# ──────────────────────────────────────────────────────────────────────────────

class OvarianPredictor:
    """
    临床预测推理器.
    接受患者基线参数 (+可选干预方案), 返回 4 个子模型的概率预测.
    """

    def __init__(self, ml_system: OvarianMLSystem):
        self.system = ml_system

    def predict(self, patient: Dict[str, Any]) -> Dict[str, float]:
        """
        对单个患者进行预测.

        Args:
            patient: 患者参数字典, 例如:
                {
                    "AMH": 1.2, "AFC": 8, "FSH": 9.5, "LH": 4.2,
                    "Age": 34, "P": 0.5, "Weight": 55, "DBP": 76,
                    "WBC": 6.1, "ALT": 25, "RBC": 4.1, "Duration": 2,
                    "POIorDOR": "No", "PCOS": "No", "PLT": 220,
                    # 干预方案 (可选, 用于策略模型)
                    "Protocol": "Long", "Initial.FSH": 150,
                    "Recombinant": "Yes", "Use.LH": "No"
                }

        Returns:
            {
                "prob_POR_dm":  ... # PORDM: 卵巢低反应概率 (诊断)
                "prob_HOR_dm":  ... # HORDM: 卵巢高反应概率 (诊断)
                "prob_POR_sm":  ... # PORSM: 卵巢低反应概率 (含干预)
                "prob_HOR_sm":  ... # HORSM: 卵巢高反应概率 (含干预)
            }
        """
        results = {}

        # ── PORDM: 诊断模型 (POR, 不含干预) ────────────────────────────────
        model_pordm = self.system.models["PORDM"]
        if model_pordm.model_ is not None:
            X_pordm = _patient_dict_to_df(
                patient,
                features=PORRM_FEATURES,
                cat_cols=["POIorDOR"]
            )
            X_pordm = model_pordm._align_features(X_pordm)
            results["prob_POR_dm"] = float(
                model_pordm.model_.predict_proba(X_pordm)[0, 1])

        # ── HORDM: 诊断模型 (HOR, 不含干预) ────────────────────────────────
        model_hordm = self.system.models["HORDM"]
        if model_hordm.model_ is not None:
            X_hordm = _patient_dict_to_df(
                patient,
                features=HORRM_FEATURES,
                cat_cols=["POIorDOR", "PCOS"]
            )
            X_hordm = model_hordm._align_features(X_hordm)
            results["prob_HOR_dm"] = float(
                model_hordm.model_.predict_proba(X_hordm)[0, 1])

        # ── PORSM: 策略模型 (POR, 含干预) ──────────────────────────────────
        model_porsm = self.system.models["PORSM"]
        if model_porsm.model_ is not None:
            X_porsm = _patient_dict_to_df(
                patient,
                features=PORRM_FEATURES + OS_INTERVENTIONS,
                cat_cols=["POIorDOR", "Protocol", "Recombinant", "Use.LH"]
            )
            X_porsm = model_porsm._align_features(X_porsm)
            results["prob_POR_sm"] = float(
                model_porsm.model_.predict_proba(X_porsm)[0, 1])

        # ── HORSM: 策略模型 (HOR, 含干预) ──────────────────────────────────
        model_horsm = self.system.models["HORSM"]
        if model_horsm.model_ is not None:
            X_horsm = _patient_dict_to_df(
                patient,
                features=HORRM_FEATURES + OS_INTERVENTIONS,
                cat_cols=["POIorDOR", "PCOS", "Protocol", "Recombinant", "Use.LH"]
            )
            X_horsm = model_horsm._align_features(X_horsm)
            results["prob_HOR_sm"] = float(
                model_horsm.model_.predict_proba(X_horsm)[0, 1])

        return results

    @classmethod
    def from_directory(cls, directory: str) -> "OvarianPredictor":
        """从保存目录加载模型."""
        system = OvarianMLSystem.load(directory)
        return cls(system)
