"""
preprocessing.py
================
数据预处理模块, 对应 R 中的 Preprocessings.Rmd

功能:
  - 读取原始 Excel 数据
  - 删除不需要的列
  - 分层训练/测试集划分 (对应 initial_split, strata=POR/HOR, prop=0.7)
  - 随机森林 MICE 缺失值填补 (对应 miceRanger)
  - one-hot 编码分类变量 (对应 step_dummy)
  - 持久化已处理的数据集
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os
from typing import Tuple, Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 列名定义
# ──────────────────────────────────────────────────────────────────────────────

# 原始数据中需要删除的列 (对应 R: select(-c(...)))
COLS_TO_DROP = [
    "ID", "Optime", "PRL", "T", "TC", "TG", "LDL", "HDL", "HCY",
    "CA125", "Insulin", "BUN", "FBG", "BMI", "A.panicillin",
    "A.cepha", "NG.DNA", "UUorMH.DNA", "Rh_neg", "Num.pretrigger"
]

# PORRM/PORSM 基线特征
PORRM_FEATURES = ["AMH", "AFC", "POIorDOR", "FSH", "Age", "P",
                  "Weight", "DBP", "WBC", "ALT", "RBC", "Duration", "LH"]

# HORRM/HORSM 基线特征
HORRM_FEATURES = ["AMH", "AFC", "FSH", "Age", "LH",
                  "POIorDOR", "PCOS", "PLT", "Weight", "Duration"]

# 4个OS干预特征 (Strategy模型额外特征)
OS_INTERVENTIONS = ["Protocol", "Initial.FSH", "Recombinant", "Use.LH"]

# 分类变量列表
CATEGORICAL_COLS = ["POIorDOR", "PCOS", "Protocol", "Recombinant", "Use.LH"]

# 目标变量
TARGET_POR = "POR"
TARGET_HOR = "HOR"


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    读取原始 Excel 文件并进行初步清洗.
    对应 R: openxlsx::read.xlsx() + mutate_if(is.character, as.factor)
    """
    df = pd.read_excel(filepath)
    # 将 Yes/No 字符串标准化
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df


def drop_unused_columns(df: pd.DataFrame,
                        extra_drop: Optional[list] = None) -> pd.DataFrame:
    """删除不需要的列, 对应 R: select(-c(...))"""
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if extra_drop:
        to_drop += [c for c in extra_drop if c in df.columns]
    return df.drop(columns=to_drop)


def stratified_split(df: pd.DataFrame,
                     target: str,
                     test_size: float = 0.3,
                     random_state: int = 777
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分层划分训练/测试集.
    对应 R: initial_split(data, strata=target, prop=0.7)
    """
    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MICE 缺失值填补 (对应 miceRanger)
# ──────────────────────────────────────────────────────────────────────────────

class MICEImputer:
    """
    基于随机森林的链式方程多重插补 (MICE).
    对应 R: miceRanger(m=1, maxiter=100, valueSelector='meanMatch', seed=777)

    使用 sklearn 的 IterativeImputer + RandomForest 实现.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 777):
        self.max_iter = max_iter
        self.random_state = random_state
        self._num_imputer: Optional[IterativeImputer] = None
        self._cat_cols: list = []
        self._num_cols: list = []
        self._cat_modes: Dict[str, any] = {}

    def fit(self, df: pd.DataFrame) -> "MICEImputer":
        """在训练数据上拟合插补器."""
        # 分离数值列和分类列
        self._cat_cols = df.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
        self._num_cols = df.select_dtypes(include="number").columns.tolist()

        # 分类列用众数填补
        for col in self._cat_cols:
            self._cat_modes[col] = df[col].mode()[0]

        # 数值列用随机森林迭代填补
        if self._num_cols:
            self._num_imputer = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            self._num_imputer.fit(df[self._num_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """填补数据集的缺失值."""
        df = df.copy()
        # 分类列: 众数填补
        for col in self._cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self._cat_modes.get(col, df[col].mode()[0]))

        # 数值列: MICE 填补
        if self._num_imputer and self._num_cols:
            available_num = [c for c in self._num_cols if c in df.columns]
            imputed = self._num_imputer.transform(df[available_num])
            df[available_num] = imputed
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并填补."""
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MICEImputer":
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# one-hot 编码 (对应 step_dummy)
# ──────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame,
                        cat_cols: Optional[list] = None,
                        drop_first: bool = True) -> pd.DataFrame:
    """
    对分类列进行 one-hot 编码.
    对应 R: step_dummy(all_nominal_predictors())
    R 默认产生 (n-1) 个哑变量, 故 drop_first=True.
    """
    cols = cat_cols or [c for c in CATEGORICAL_COLS if c in df.columns]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return pd.get_dummies(df, columns=present, drop_first=drop_first,
                          dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 完整预处理管道
# ──────────────────────────────────────────────────────────────────────────────

class OvarianPreprocessor:
    """
    封装完整预处理管道:
      1. 删除无用列
      2. 分层训练/测试集划分
      3. MICE 填补 (分别对 POR/HOR 数据集)
      4. one-hot 编码
    """

    def __init__(self, random_state: int = 777):
        self.random_state = random_state
        self.porsm_imputer = MICEImputer(random_state=random_state)
        self.horsm_imputer = MICEImputer(random_state=random_state)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _select_porsm(self, df: pd.DataFrame) -> pd.DataFrame:
        """选取 PORSM 所需列."""
        cols = PORRM_FEATURES + OS_INTERVENTIONS + [TARGET_POR]
        return df[[c for c in cols if c in df.columns]].copy()

    def _select_horsm(self, df: pd.DataFrame) -> pd.DataFrame:
        """选取 HORSM 所需列."""
        cols = HORRM_FEATURES + OS_INTERVENTIONS + [TARGET_HOR]
        return df[[c for c in cols if c in df.columns]].copy()

    def _select_pordm(self, df: pd.DataFrame) -> pd.DataFrame:
        """选取 PORDM (无干预特征) 所需列."""
        cols = PORRM_FEATURES + [TARGET_POR]
        return df[[c for c in cols if c in df.columns]].copy()

    def _select_hordm(self, df: pd.DataFrame) -> pd.DataFrame:
        """选取 HORDM (无干预特征) 所需列."""
        cols = HORRM_FEATURES + [TARGET_HOR]
        return df[[c for c in cols if c in df.columns]].copy()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def fit_transform(self, raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        对原始数据集执行完整预处理.

        返回字典:
          {
            'porsm_train', 'porsm_test',
            'horsm_train', 'horsm_test',
            'pordm_train', 'pordm_test',
            'hordm_train', 'hordm_test',
          }
        """
        df = drop_unused_columns(raw_df)

        # --- POR 方向 ---
        porsm_df = self._select_porsm(df)
        # 确保目标变量是字符串 Yes/No (与 R 一致)
        porsm_df[TARGET_POR] = porsm_df[TARGET_POR].astype(str)
        porsm_train_raw, porsm_test_raw = stratified_split(
            porsm_df, TARGET_POR, random_state=self.random_state)

        porsm_train = self.porsm_imputer.fit_transform(porsm_train_raw)
        porsm_test = self.porsm_imputer.transform(porsm_test_raw)

        porsm_train = encode_categoricals(porsm_train)
        porsm_test = encode_categoricals(porsm_test)

        # --- HOR 方向 ---
        horsm_df = self._select_horsm(df)
        horsm_df[TARGET_HOR] = horsm_df[TARGET_HOR].astype(str)
        horsm_train_raw, horsm_test_raw = stratified_split(
            horsm_df, TARGET_HOR, random_state=self.random_state)

        horsm_train = self.horsm_imputer.fit_transform(horsm_train_raw)
        horsm_test = self.horsm_imputer.transform(horsm_test_raw)

        horsm_train = encode_categoricals(horsm_train)
        horsm_test = encode_categoricals(horsm_test)

        # --- 诊断模型 (去掉干预特征) ---
        os_cols_dummy = [c for c in porsm_train.columns
                         if any(c.startswith(p) for p in OS_INTERVENTIONS)]
        pordm_train = porsm_train.drop(columns=os_cols_dummy, errors="ignore")
        pordm_test = porsm_test.drop(columns=os_cols_dummy, errors="ignore")

        os_cols_dummy_h = [c for c in horsm_train.columns
                           if any(c.startswith(p) for p in OS_INTERVENTIONS)]
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
        """持久化插补器."""
        os.makedirs(directory, exist_ok=True)
        self.porsm_imputer.save(os.path.join(directory, "porsm_imputer.pkl"))
        self.horsm_imputer.save(os.path.join(directory, "horsm_imputer.pkl"))

    @classmethod
    def load(cls, directory: str) -> "OvarianPreprocessor":
        proc = cls.__new__(cls)
        proc.porsm_imputer = MICEImputer.load(
            os.path.join(directory, "porsm_imputer.pkl"))
        proc.horsm_imputer = MICEImputer.load(
            os.path.join(directory, "horsm_imputer.pkl"))
        return proc


# ──────────────────────────────────────────────────────────────────────────────
# 工具: 生成合成测试数据集 (无真实数据时使用)
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_dataset(n: int = 1000, random_state: int = 777) -> pd.DataFrame:
    """
    生成用于测试流程的合成数据集.
    特征分布参考文献中的正常范围.
    """
    rng = np.random.default_rng(random_state)

    n_por = int(n * 0.22)   # ~22% POR 发生率
    n_hor = int(n * 0.15)   # ~15% HOR 发生率

    df = pd.DataFrame({
        # 连续变量
        "AMH":      np.clip(rng.lognormal(0.5, 0.8, n), 0.01, 20),
        "AFC":      np.clip(rng.poisson(10, n).astype(float), 1, 40),
        "FSH":      np.clip(rng.normal(7, 3, n), 1, 30),
        "LH":       np.clip(rng.normal(5, 2, n), 0.5, 20),
        "Age":      np.clip(rng.normal(32, 5, n), 20, 45),
        "P":        np.clip(rng.lognormal(-1, 0.5, n), 0.01, 2),
        "Weight":   np.clip(rng.normal(58, 10, n), 35, 100),
        "DBP":      np.clip(rng.normal(76, 10, n), 50, 110),
        "WBC":      np.clip(rng.normal(6, 1.5, n), 2, 15),
        "ALT":      np.clip(rng.lognormal(2.8, 0.4, n), 5, 100),
        "RBC":      np.clip(rng.normal(4.2, 0.4, n), 2.5, 6),
        "Duration": np.clip(rng.exponential(2, n), 0.5, 15),
        "PLT":      np.clip(rng.normal(220, 50, n), 50, 450),
        # 分类变量
        "POIorDOR": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
        "PCOS":     rng.choice(["Yes", "No"], n, p=[0.12, 0.88]),
        "Protocol": rng.choice(["Long", "Short", "Antagonist"], n,
                               p=[0.5, 0.2, 0.3]),
        "Initial.FSH": np.clip(rng.normal(150, 50, n), 75, 300),
        "Recombinant": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "Use.LH":      rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "Num.oocytes": np.clip(rng.poisson(10, n).astype(float), 0, 40),
    })

    # 基于 AMH/AFC/FSH/Age 构建 POR/HOR 标签
    por_score = (-df["AMH"] * 0.5 - df["AFC"] * 0.3
                 + df["FSH"] * 0.4 + df["Age"] * 0.06
                 + (df["POIorDOR"] == "Yes").astype(float) * 1.5)
    por_prob = 1 / (1 + np.exp(-por_score + por_score.median()))
    df["POR"] = (rng.random(n) < por_prob * 0.22 / por_prob.mean()
                 ).map({True: "Yes", False: "No"})

    hor_score = (df["AMH"] * 0.6 + df["AFC"] * 0.4
                 - df["FSH"] * 0.2 - df["Age"] * 0.05
                 + (df["PCOS"] == "Yes").astype(float) * 1.2)
    hor_prob = 1 / (1 + np.exp(-hor_score + hor_score.median()))
    df["HOR"] = (rng.random(n) < hor_prob * 0.15 / hor_prob.mean()
                 ).map({True: "Yes", False: "No"})

    # 随机引入约 10% 缺失值 (AMH/AFC 最常见)
    for col, miss_rate in [("AMH", 0.12), ("AFC", 0.08), ("LH", 0.05),
                           ("P", 0.06), ("ALT", 0.04)]:
        mask = rng.random(n) < miss_rate
        df.loc[mask, col] = np.nan

    return df
