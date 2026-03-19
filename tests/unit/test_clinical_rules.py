from ovarian_prediction.clinical import risk_level, suggest_fsh_dose
from ovarian_prediction.clinical.thresholds import POR_THRESHOLDS


def test_risk_level_thresholds_are_applied():
    assert risk_level(0.1, POR_THRESHOLDS) == "低"
    assert risk_level(0.3, POR_THRESHOLDS) == "中"
    assert risk_level(0.6, POR_THRESHOLDS) == "高"


def test_suggest_fsh_dose_respects_risk_direction():
    high_por = suggest_fsh_dose(0.7, 0.1, weight=55)
    high_hor = suggest_fsh_dose(0.1, 0.6, weight=55)
    assert "IU/天" in high_por
    assert "IU/天" in high_hor
    assert high_por != high_hor
