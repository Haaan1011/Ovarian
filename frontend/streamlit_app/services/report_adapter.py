from typing import Dict, Optional


def reserve_probability(report: Optional[Dict]) -> float:
    if not report:
        return 0.0
    return float(report["risk_levels"]["POR"]["prob"])


def hor_probability(report: Optional[Dict]) -> float:
    if not report:
        return 0.0
    return float(report["risk_levels"]["HOR"]["prob"])
