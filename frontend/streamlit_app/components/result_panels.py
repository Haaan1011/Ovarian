from html import escape
from textwrap import dedent
from typing import Dict, List, Optional, Union

from ovarian_prediction.clinical import HOR_THRESHOLDS, POR_THRESHOLDS

from frontend.streamlit_app.components.metric_cards import (
    clinical_metric_card,
    metric_box,
    svg_circular_progress_html,
)
from frontend.streamlit_app.services.reference_api import (
    age_based_embryo_aneuploidy,
    estimate_reserve_reference,
)
from frontend.streamlit_app.services.report_adapter import hor_probability, reserve_probability
from frontend.streamlit_app.utils.formatting import fmt_value


def reserve_profile(por_prob: float) -> Dict[str, Union[float, str]]:
    score = max(0.0, min(100.0, 100.0 - por_prob * 100.0))
    if score >= 75:
        title = "储备功能较好"
    elif score >= 55:
        title = "储备功能中等"
    else:
        title = "储备功能偏低"
    return {"score": score, "title": title, "accent": "#df5b57"}


def reserve_grade(score: float) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    score = max(0.0, min(100.0, float(score)))
    levels = [
        {"letter": "A", "label": "A级", "title": "储备功能优秀", "accent": "#96d66a", "threshold": 88},
        {"letter": "B", "label": "B级", "title": "储备功能较好", "accent": "#e0d85f", "threshold": 75},
        {"letter": "C", "label": "C级", "title": "储备功能中等", "accent": "#f2a276", "threshold": 55},
        {"letter": "D", "label": "D级", "title": "储备功能偏低", "accent": "#ef6463", "threshold": 0},
    ]
    current = levels[-1]
    for item in levels:
        if score >= int(item["threshold"]):
            current = item
            break
    return {
        "letter": str(current["letter"]),
        "label": str(current["label"]),
        "title": str(current["title"]),
        "accent": str(current["accent"]),
        "levels": levels,
    }


def ohss_profile(hor_prob: float) -> Dict[str, Union[float, str]]:
    percent = max(0.0, min(100.0, hor_prob * 100.0))
    if hor_prob < HOR_THRESHOLDS["low"]:
        level = "OHSS 风险较低"
    elif hor_prob < HOR_THRESHOLDS["high"]:
        level = "OHSS 风险中等"
    else:
        level = "OHSS 风险较高"
    return {"percent": percent, "level": level}


def reserve_result_html(report: Optional[Dict], patient: Optional[Dict]) -> str:
    if not report or not patient:
        return (
            "<div class='result-zone'><div class='result-card reserve-result-card placeholder-card'>"
            "录入左侧指标后，这里生成卵巢储备功能百分比结果。"
            "</div></div>"
        )

    por_prob = reserve_probability(report)
    profile = reserve_profile(por_prob)
    filled_percent = max(0.0, min(100.0, float(profile["score"])))
    grade = reserve_grade(filled_percent)
    reference = patient.get("reserve_reference") or estimate_reserve_reference(
        int(patient.get("Age") or 32),
        float(profile["score"]),
    )
    embryo = patient.get("embryo_aneuploidy") or age_based_embryo_aneuploidy(float(patient.get("Age") or 32))
    source_text = "公开工具参考" if reference.get("mode") == "reference" else "本地趋势估算"
    grade_list_html = "".join(
        (
            "<div class='reserve-grade-item"
            + (" active" if str(item["letter"]) == str(grade["letter"]) else "")
            + "'>"
            + f"<span class='reserve-grade-swatch' style='background:{escape(str(item['accent']))};'></span>"
            + "<div class='reserve-grade-item-text'>"
            + f"<span class='reserve-grade-item-label'>{escape(str(item['label']))}</span>"
            + f"<span class='reserve-grade-item-title'>{escape(str(item['title']))}</span>"
            + "</div>"
            + "</div>"
        )
        for item in grade["levels"]
    )

    return dedent(
        f"""
        <div class='result-zone'>
            <div class='result-card reserve-result-card'>
                <div class='card-label'>Ovarian Reserve Output</div>
                <div class='card-title'>卵巢储备功能结果</div>
                <div class='reserve-layout'>
                    <div class='reserve-score-card'>
                        <div class='reserve-score-layout'>
                            <div class='reserve-score-visual'>
                                <div class='reserve-ring-wrap'>
                                    {svg_circular_progress_html(filled_percent, size=252, stroke_width=22, caption="Reserve Index")}
                                </div>
                            </div>
                            <div class='reserve-grade-panel'>
                                <div class='reserve-grade-label'>级别评级</div>
                                <div class='reserve-grade-current'>
                                    <div class='reserve-grade-meta'>
                                        <div class='reserve-grade-letter' style='color:{escape(str(grade["accent"]))};'>{escape(str(grade["letter"]))}</div>
                                        <div class='reserve-grade-copy'>
                                            <div class='reserve-grade-name'>{escape(str(grade["label"]))}</div>
                                            <div class='reserve-grade-conclusion'>{escape(str(profile["title"]))}</div>
                                            <div class='reserve-grade-note'>当前储备指数位于该评级区间，评级结果与环形百分比同步对应。</div>
                                        </div>
                                    </div>
                                </div>
                                <div class='reserve-grade-list'>{grade_list_html}</div>
                            </div>
                        </div>
                    </div>
                    <div class='reserve-copy'>
                        <div class='reserve-headline'>预测结果</div>
                        <div class='reserve-clinical-grid'>
                            {clinical_metric_card("Starting age of Diminished Ovarian Reserve", "卵巢储备减退起始年龄", f"{float(reference['dor_age']):.1f} 岁", "表示储备功能进入明显下降阶段的参考起点。", f"区间 [{float(reference['dor_age_lower']):.1f}, {float(reference['dor_age_upper']):.1f}]")}
                            {clinical_metric_card("Starting age of peri-menopause", "围绝经期起始年龄", f"{float(reference['peri_age']):.1f} 岁", "表示围绝经期变化开始出现的参考年龄节点。", f"区间 [{float(reference['peri_age_lower']):.1f}, {float(reference['peri_age_upper']):.1f}]")}
                            {clinical_metric_card("Age-related embryo aneuploidy", "按年龄估算的胚胎染色体异常率", f"{float(embryo['risk']):.1f}%", "基于女性年龄的趋势估算，不等同于实际 PGT-A 检测结果。", f"预估整倍体胚胎比例 {float(embryo['euploid']):.1f}%", wide=True, badges=[f"风险分层 {escape(str(embryo['label']))}", f"年龄 {fmt_value(patient.get('Age'))} 岁", source_text])}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()


def plan_result_html(report: Optional[Dict], patient: Optional[Dict]) -> str:
    if not report or not patient:
        return (
            "<div class='result-zone'><div class='plan-stack'>"
            "<div class='result-card mini-card placeholder-card'>录入右侧指标后，这里显示 OHSS 风险百分比。</div>"
            "<div class='result-card mini-card placeholder-card'>生成后，这里显示获卵数、推荐方案、起始 FSH、LH 支持与 FSH 类型。</div>"
            "</div></div>"
        )

    hor_prob = hor_probability(report)
    profile = ohss_profile(hor_prob)
    recs = report["recommendations"]
    metrics = [
        metric_box("预期获卵数", recs.get("est_oocytes") or "--", "基于 AMH / AFC / 年龄估算"),
        metric_box("推荐方案", recs["protocol"], "当前更匹配的促排方向"),
        metric_box("起始 FSH", recs["fsh_dose"], "推荐起始剂量区间"),
        metric_box("LH 支持", recs["lh_support"], "是否考虑外源性 LH"),
        metric_box("FSH 类型", recs["fsh_type"], "药物来源建议"),
        metric_box("录入摘要", f"年龄 {fmt_value(patient.get('Age'))} / BMI {fmt_value(patient.get('BMI'), 1)} / 周期 {fmt_value(patient.get('cycle_length'))}", "月经周期与 BMI 当前仅作病例记录"),
    ]
    return dedent(
        f"""
        <div class='result-zone'>
            <div class='plan-stack'>
                <div class='result-card mini-card'>
                    <div class='card-label'>Ovulation Stimulation Plan</div>
                    <div class='card-title'>促排卵方案</div>
                    <div class='ohss-shell'>
                        <div class='ohss-header'>
                            <div>
                                <div class='ohss-value'>{profile["percent"]:.1f}%</div>
                                <div class='ohss-level'>{escape(str(profile["level"]))}</div>
                            </div>
                        </div>
                        <div class='risk-bar'><div class='risk-bar-fill' style='width:{profile["percent"]:.1f}%'></div></div>
                        <div class='mini-note'>当前后端模型直接使用年龄、AMH、AFC、FSH、LH；月经周期和 BMI 暂未作为模型特征参与计算。</div>
                    </div>
                </div>
                <div class='result-card mini-card'>
                    <div class='card-label'>Stimulation Plan</div>
                    <div class='card-title'>方案建议</div>
                    <div class='metric-grid'>{"".join(metrics)}</div>
                </div>
            </div>
        </div>
        """
    ).strip()
