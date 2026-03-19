from typing import Any, Dict, Optional

from .rules import risk_emoji


def make_prob_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}|"


def format_clinical_report(
    patient: Dict[str, Any],
    probs: Dict[str, float],
    por_prob: float,
    hor_prob: float,
    por_level: str,
    hor_level: str,
    recs: Dict[str, Any],
    por_prob_dm: Optional[float],
    hor_prob_dm: Optional[float],
) -> str:
    sep = "─" * 56
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║       卵巢反应预测及促排卵策略推荐系统              ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
        "【患者基线参数】",
        f"  年龄: {patient.get('Age', 'N/A')} 岁   体重: {patient.get('Weight', 'N/A')} kg   不孕年限: {patient.get('Duration', 'N/A')} 年",
        f"  AMH:  {patient.get('AMH', 'N/A')} ng/mL   AFC: {patient.get('AFC', 'N/A')} 个   基础FSH: {patient.get('FSH', 'N/A')} mIU/mL",
        f"  LH:   {patient.get('LH', 'N/A')} mIU/mL   P: {patient.get('P', 'N/A')} ng/mL",
        f"  POI/DOR诊断: {patient.get('POIorDOR', 'N/A')}   PCOS: {patient.get('PCOS', 'N/A')}",
        "",
        sep,
        "【🔬 风险预测结果】",
        sep,
        "  卵巢低反应 (POR/DOR) 风险:",
        f"    {risk_emoji(por_level)} 风险等级: {por_level}风险",
        f"    概率: {por_prob * 100:.1f}%  {make_prob_bar(por_prob)}",
    ]
    if por_prob_dm is not None and abs(por_prob_dm - por_prob) > 0.01:
        lines.append(f"    (基础预测: {por_prob_dm*100:.1f}%  含方案预测: {por_prob*100:.1f}%)")
    lines.extend(
        [
            "",
            "  卵巢高反应 (OHSS/HOR) 风险:",
            f"    {risk_emoji(hor_level)} 风险等级: {hor_level}风险",
            f"    概率: {hor_prob * 100:.1f}%  {make_prob_bar(hor_prob)}",
        ]
    )
    if hor_prob_dm is not None and abs(hor_prob_dm - hor_prob) > 0.01:
        lines.append(f"    (基础预测: {hor_prob_dm*100:.1f}%  含方案预测: {hor_prob*100:.1f}%)")
    lines.append("")

    if recs.get("est_oocytes"):
        lines.append(f"  预期获卵数估计: {recs['est_oocytes']}")
    lines.extend(
        [
            "",
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
    )
    return "\n".join(lines)
