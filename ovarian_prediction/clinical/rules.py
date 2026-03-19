from typing import Dict, Optional

from .thresholds import FSH_DOSE_BY_RISK, HOR_THRESHOLDS, POR_THRESHOLDS


def risk_level(prob: float, thresholds: Dict[str, float]) -> str:
    if prob < thresholds["low"]:
        return "低"
    if prob < thresholds["high"]:
        return "中"
    return "高"


def risk_emoji(level: str) -> str:
    return {"低": "🟢", "中": "🟡", "高": "🔴"}.get(level, "⚪")


def estimate_oocyte_count(
    amh: Optional[float],
    afc: Optional[float],
    age: Optional[float],
) -> Optional[str]:
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


def suggest_protocol(
    amh: Optional[float],
    afc: Optional[float],
    fsh: Optional[float],
    por_prob: float,
    hor_prob: float,
) -> str:
    del amh, afc, fsh
    por_level = risk_level(por_prob, POR_THRESHOLDS)
    hor_level = risk_level(hor_prob, HOR_THRESHOLDS)

    if por_level == "高":
        return "拮抗剂方案 (Antagonist) 或微刺激方案"
    if por_level == "中":
        return "拮抗剂方案 (Antagonist) 或长效长方案"
    if hor_level == "高":
        return "拮抗剂方案 (Antagonist) ⚠️ 注意OHSS风险"
    if hor_level == "中":
        return "拮抗剂方案 (Antagonist)"
    return "长方案 (Long GnRH-agonist) 或拮抗剂方案"


def suggest_fsh_dose(
    por_prob: float,
    hor_prob: float,
    weight: Optional[float] = None,
) -> str:
    por_level = risk_level(por_prob, POR_THRESHOLDS)
    hor_level = risk_level(hor_prob, HOR_THRESHOLDS)

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

    weight_note = ""
    if weight is not None:
        if weight > 70:
            low += 25
            high += 25
            weight_note = "（已根据体重上调）"
        elif weight < 50:
            low -= 25
            high -= 25
            weight_note = "（已根据体重下调）"

    return f"{max(37, low)} ~ {min(450, high)} IU/天 {weight_note}"


def suggest_fsh_type(por_prob: float) -> str:
    por_level = risk_level(por_prob, POR_THRESHOLDS)
    if por_level in ("中", "高"):
        return "推荐使用重组FSH (rFSH) 以获得更精确的剂量控制"
    return "重组FSH (rFSH) 或尿源性FSH (uFSH) 均可"


def suggest_lh(por_prob: float, age: Optional[float]) -> str:
    por_level = risk_level(por_prob, POR_THRESHOLDS)
    age_flag = age is not None and age >= 35
    if por_level == "高" or age_flag:
        return "建议添加外源性LH（rLH 或 hMG）以支持卵泡发育"
    if por_level == "中":
        return "可考虑在卵泡晚期适当添加外源性LH"
    return "通常无需额外补充LH"
