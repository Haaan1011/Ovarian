"""
clinical_system.py
==================
临床友好型决策系统（核心对外接口）

功能:
  - 接受患者基线参数（模拟 R Shiny 网页应用的输入）
  - 调用 4 个子模型获取概率预测
  - 生成临床语言的风险分层报告
  - 提供促排卵策略建议 (Protocol/FSH剂量/rFSH/LH补充)
  - 支持探索不同干预方案的"模拟比较"

风险阈值参考原文献:
  POR: prob > 0.30 → 中风险; > 0.50 → 高风险
  HOR: prob > 0.20 → 中风险; > 0.40 → 高风险
"""

import numpy as np
from typing import Dict, Any, Optional, List
import textwrap

from .predict import OvarianPredictor
from .models import OvarianMLSystem


# ──────────────────────────────────────────────────────────────────────────────
# 风险阈值 & 临床建议
# ──────────────────────────────────────────────────────────────────────────────

POR_THRESHOLDS = {"low": 0.20, "high": 0.40}
HOR_THRESHOLDS = {"low": 0.20, "high": 0.35}

# 预期卵母细胞数参考范围
OOCYTE_NORMAL_RANGE = (4, 20)

# 4种促排方案选项（与R数据集一致）
PROTOCOLS = ["Long", "Short", "Antagonist"]

# 典型FSH起始剂量范围 (IU/day)
FSH_DOSE_BY_RISK = {
    "POR_high": (225, 300),
    "POR_low":  (150, 225),
    "normal":   (100, 150),
    "HOR_low":  (75, 112),
    "HOR_high": (37, 75),
}


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _risk_level(prob: float, thresholds: Dict[str, float]) -> str:
    """将概率映射为 '低/中/高' 风险等级."""
    if prob < thresholds["low"]:
        return "低"
    elif prob < thresholds["high"]:
        return "中"
    else:
        return "高"


def _risk_emoji(level: str) -> str:
    return {"低": "🟢", "中": "🟡", "高": "🔴"}.get(level, "⚪")


def _estimate_oocyte_count(amh: Optional[float],
                           afc: Optional[float],
                           age: Optional[float]) -> Optional[str]:
    """
    基于 AMH/AFC/年龄简单估计预期卵母细胞数区间.
    参考文献阈值: AMH<1.1 → DOR; AMH>3.4 → 高反应
    """
    if amh is None and afc is None:
        return None
    score = 0.0
    if amh is not None:
        score += amh * 1.5
    if afc is not None:
        score += afc * 0.4
    if age is not None:
        score -= (age - 30) * 0.3

    low = max(1, round(score * 0.7))
    high = max(low + 1, round(score * 1.3) + 3)
    return f"{low} ~ {high} 枚"


def _suggest_protocol(amh: Optional[float],
                      afc: Optional[float],
                      fsh: Optional[float],
                      por_prob: float,
                      hor_prob: float) -> str:
    """
    根据预测概率和基线参数推荐促排方案.
    参考 Bologna 标准和国内指南.
    """
    por_level = _risk_level(por_prob, POR_THRESHOLDS)
    hor_level = _risk_level(hor_prob, HOR_THRESHOLDS)

    if por_level == "高":
        return "拮抗剂方案 (Antagonist) 或微刺激方案"
    elif por_level == "中":
        return "拮抗剂方案 (Antagonist) 或长效长方案"
    elif hor_level == "高":
        return "拮抗剂方案 (Antagonist) ⚠️ 注意OHSS风险"
    elif hor_level == "中":
        return "拮抗剂方案 (Antagonist)"
    else:
        return "长方案 (Long GnRH-agonist) 或拮抗剂方案"


def _suggest_fsh_dose(por_prob: float,
                      hor_prob: float,
                      weight: Optional[float] = None) -> str:
    """FSH起始剂量建议."""
    por_level = _risk_level(por_prob, POR_THRESHOLDS)
    hor_level = _risk_level(hor_prob, HOR_THRESHOLDS)

    if por_level == "高":
        low, high = FSH_DOSE_BY_RISK["POR_high"]
    elif por_level == "中":
        low, high = FSH_DOSE_BY_RISK["POR_low"]
    elif hor_level == "高":
        low, high = FSH_DOSE_BY_RISK["HOR_high"]
    elif hor_level == "中":
        low, high = FSH_DOSE_BY_RISK["HOR_low"]
    else:
        low, high = FSH_DOSE_BY_RISK["normal"]

    # 体重修正 (+/-25 IU)
    weight_note = ""
    if weight is not None:
        if weight > 70:
            low += 25; high += 25
            weight_note = "（已根据体重上调）"
        elif weight < 50:
            low -= 25; high -= 25
            weight_note = "（已根据体重下调）"

    return f"{max(37, low)} ~ {min(450, high)} IU/天 {weight_note}"


def _suggest_fsh_type(por_prob: float) -> str:
    """rFSH vs uFSH 建议."""
    por_level = _risk_level(por_prob, POR_THRESHOLDS)
    if por_level in ("中", "高"):
        return "推荐使用重组FSH (rFSH) 以获得更精确的剂量控制"
    return "重组FSH (rFSH) 或尿源性FSH (uFSH) 均可"


def _suggest_lh(por_prob: float, age: Optional[float]) -> str:
    """外源性LH补充建议."""
    por_level = _risk_level(por_prob, POR_THRESHOLDS)
    age_flag = age is not None and age >= 35

    if por_level == "高" or age_flag:
        return "建议添加外源性LH（rLH 或 hMG）以支持卵泡发育"
    elif por_level == "中":
        return "可考虑在卵泡晚期适当添加外源性LH"
    return "通常无需额外补充LH"


# ──────────────────────────────────────────────────────────────────────────────
# 核心决策系统
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalDecisionSystem:
    """
    临床友好型卵巢反应预测及促排策略推荐系统.

    使用示例:
        system = ClinicalDecisionSystem.from_directory("models/")
        report = system.evaluate_patient({
            "AMH": 1.2, "AFC": 8, "Age": 34, "FSH": 9.5, ...
        })
        print(report["summary"])
    """

    def __init__(self, predictor: OvarianPredictor):
        self.predictor = predictor

    @classmethod
    def from_directory(cls, directory: str) -> "ClinicalDecisionSystem":
        """从已保存的模型目录构建系统."""
        predictor = OvarianPredictor.from_directory(directory)
        return cls(predictor)

    @classmethod
    def from_ml_system(cls, system: OvarianMLSystem) -> "ClinicalDecisionSystem":
        """从 OvarianMLSystem 对象构建."""
        predictor = OvarianPredictor(system)
        return cls(predictor)

    # ── 主接口 ──────────────────────────────────────────────────────────────────

    def evaluate_patient(self,
                         patient: Dict[str, Any],
                         intervention: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
        """
        核心评估接口.

        Args:
            patient: 患者基线参数字典 (必填字段见下)
            intervention: 可选干预方案参数，若提供则同时运行策略模型
                {
                    "Protocol": "Long" | "Short" | "Antagonist",
                    "Initial.FSH": 150,          # IU/day
                    "Recombinant": "Yes" | "No", # 是否rFSH
                    "Use.LH": "Yes" | "No"        # 是否补充LH
                }

        Returns:
            dict 包含:
              - raw_probs: 原始概率值
              - risk_levels: 风险等级
              - recommendations: 促排建议字典
              - summary: 格式化的中文报告字符串
        """
        # 合并干预参数
        full_patient = {**patient}
        if intervention:
            full_patient.update(intervention)

        # 调用预测器
        probs = self.predictor.predict(full_patient)

        # 提取概率 (优先用策略模型, fallback 诊断模型)
        por_prob = probs.get("prob_POR_sm", probs.get("prob_POR_dm", 0.0))
        hor_prob = probs.get("prob_HOR_sm", probs.get("prob_HOR_dm", 0.0))
        por_prob_dm = probs.get("prob_POR_dm")
        hor_prob_dm = probs.get("prob_HOR_dm")

        por_level = _risk_level(por_prob, POR_THRESHOLDS)
        hor_level = _risk_level(hor_prob, HOR_THRESHOLDS)

        # 临床建议
        amh = patient.get("AMH")
        afc = patient.get("AFC")
        age = patient.get("Age")
        weight = patient.get("Weight")

        recommendations = {
            "protocol":    _suggest_protocol(amh, afc, patient.get("FSH"),
                                             por_prob, hor_prob),
            "fsh_dose":    _suggest_fsh_dose(por_prob, hor_prob, weight),
            "fsh_type":    _suggest_fsh_type(por_prob),
            "lh_support":  _suggest_lh(por_prob, age),
            "est_oocytes": _estimate_oocyte_count(amh, afc, age),
        }

        result = {
            "raw_probs": probs,
            "risk_levels": {
                "POR": {"prob": por_prob, "level": por_level},
                "HOR": {"prob": hor_prob, "level": hor_level},
            },
            "recommendations": recommendations,
            "summary": self._format_report(
                patient, probs, por_prob, hor_prob,
                por_level, hor_level, recommendations,
                por_prob_dm, hor_prob_dm
            )
        }
        return result

    def compare_interventions(self,
                              patient: Dict[str, Any],
                              interventions: List[Dict[str, Any]]
                              ) -> List[Dict[str, Any]]:
        """
        对同一患者比较多个促排方案, 输出每个方案的预测结果.
        适用于临床决策中对不同方案的模拟比较.
        """
        results = []
        for intv in interventions:
            result = self.evaluate_patient(patient, intv)
            result["intervention"] = intv
            results.append(result)
        return results

    # ── 报告格式化 ──────────────────────────────────────────────────────────────

    def _format_report(self,
                       patient, probs,
                       por_prob, hor_prob,
                       por_level, hor_level,
                       recs,
                       por_prob_dm, hor_prob_dm) -> str:
        """生成中文格式化临床报告."""

        sep = "─" * 56
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║       卵巢反应预测及促排卵策略推荐系统              ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            "【患者基线参数】",
            f"  年龄: {patient.get('Age', 'N/A')} 岁   "
            f"体重: {patient.get('Weight', 'N/A')} kg   "
            f"不孕年限: {patient.get('Duration', 'N/A')} 年",
            f"  AMH:  {patient.get('AMH', 'N/A')} ng/mL   "
            f"AFC: {patient.get('AFC', 'N/A')} 个   "
            f"基础FSH: {patient.get('FSH', 'N/A')} mIU/mL",
            f"  LH:   {patient.get('LH', 'N/A')} mIU/mL   "
            f"P: {patient.get('P', 'N/A')} ng/mL",
            f"  POI/DOR诊断: {patient.get('POIorDOR', 'N/A')}   "
            f"PCOS: {patient.get('PCOS', 'N/A')}",
            "",
            sep,
            "【🔬 风险预测结果】",
            sep,
        ]

        # POR 风险
        por_bar = _make_prob_bar(por_prob)
        lines += [
            f"  卵巢低反应 (POR/DOR) 风险:",
            f"    {_risk_emoji(por_level)} 风险等级: {por_level}风险",
            f"    概率: {por_prob * 100:.1f}%  {por_bar}",
        ]
        if por_prob_dm is not None and abs(por_prob_dm - por_prob) > 0.01:
            lines.append(f"    (基础预测: {por_prob_dm*100:.1f}%  含方案预测: {por_prob*100:.1f}%)")
        lines.append("")

        # HOR 风险
        hor_bar = _make_prob_bar(hor_prob)
        lines += [
            f"  卵巢高反应 (OHSS/HOR) 风险:",
            f"    {_risk_emoji(hor_level)} 风险等级: {hor_level}风险",
            f"    概率: {hor_prob * 100:.1f}%  {hor_bar}",
        ]
        if hor_prob_dm is not None and abs(hor_prob_dm - hor_prob) > 0.01:
            lines.append(f"    (基础预测: {hor_prob_dm*100:.1f}%  含方案预测: {hor_prob*100:.1f}%)")
        lines.append("")

        if recs.get("est_oocytes"):
            lines.append(f"  预期获卵数估计: {recs['est_oocytes']}")
        lines.append("")

        # 促排建议
        lines += [
            sep,
            "【💊 促排卵方案建议】",
            sep,
            f"  1. 促排方案:    {recs['protocol']}",
            f"  2. FSH起始剂量: {recs['fsh_dose']}",
            f"  3. FSH类型:     {recs['fsh_type']}",
            f"  4. LH支持:      {recs['lh_support']}",
            "",
            sep,
            "⚠️  本系统仅供临床参考, 具体治疗方案请结合患者实际情况",
            "   由经验丰富的生殖专科医师综合判断. ",
            sep,
        ]

        return "\n".join(lines)


def _make_prob_bar(prob: float, width: int = 20) -> str:
    """生成简单的 ASCII 概率条."""
    filled = round(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}|"


# ──────────────────────────────────────────────────────────────────────────────
# CLI 演示入口
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(ml_system: Optional[OvarianMLSystem] = None) -> None:
    """
    运行演示: 使用预训练模型或合成数据模型预测典型患者.
    """
    from .preprocessing import make_synthetic_dataset, OvarianPreprocessor
    from .models import OvarianMLSystem as MLS

    print("╔══════════════════════════════════════════════════════╗")
    print("║           卵巢反应预测系统 - 演示模式               ║")
    print("╚══════════════════════════════════════════════════════╝")

    if ml_system is None:
        print("\n[INFO] 未检测到预训练模型，使用合成数据快速训练演示模型...")
        df = make_synthetic_dataset(n=500, random_state=777)
        proc = OvarianPreprocessor()
        data = proc.fit_transform(df)
        ml_system = MLS(n_trials=10)
        ml_system.train_all(data, tune=False)   # tune=False: 快速演示

    cds = ClinicalDecisionSystem.from_ml_system(ml_system)

    # 典型患者 1: 疑似 DOR/POR 患者
    patient_por = {
        "AMH": 0.5,   # 低AMH
        "AFC": 4,     # 低AFC
        "FSH": 14.0,  # 高基础FSH
        "LH": 6.5,
        "Age": 38,
        "P": 0.4,
        "Weight": 52,
        "DBP": 78,
        "WBC": 5.8,
        "ALT": 22,
        "RBC": 3.9,
        "Duration": 3,
        "POIorDOR": "Yes",
        "PCOS": "No",
        "PLT": 195,
    }
    intervention_suggested = {
        "Protocol": "Antagonist",
        "Initial.FSH": 225,
        "Recombinant": "Yes",
        "Use.LH": "Yes",
    }
    print("\n" + "="*58)
    print("  📋 典型患者示例 1: 疑似DOR/卵巢低反应患者")
    print("="*58)
    result1 = cds.evaluate_patient(patient_por, intervention_suggested)
    print(result1["summary"])

    # 典型患者 2: 疑似 OHSS/高反应患者
    patient_hor = {
        "AMH": 6.8,   # 高AMH
        "AFC": 22,    # 高AFC
        "FSH": 5.1,
        "LH": 4.8,
        "Age": 27,
        "P": 0.3,
        "Weight": 55,
        "DBP": 72,
        "WBC": 6.2,
        "ALT": 18,
        "RBC": 4.3,
        "Duration": 1,
        "POIorDOR": "No",
        "PCOS": "Yes",
        "PLT": 260,
    }
    intervention_hos = {
        "Protocol": "Antagonist",
        "Initial.FSH": 75,
        "Recombinant": "Yes",
        "Use.LH": "No",
    }
    print("\n" + "="*58)
    print("  📋 典型患者示例 2: PCOS/疑似高反应患者")
    print("="*58)
    result2 = cds.evaluate_patient(patient_hor, intervention_hos)
    print(result2["summary"])


if __name__ == "__main__":
    run_demo()
