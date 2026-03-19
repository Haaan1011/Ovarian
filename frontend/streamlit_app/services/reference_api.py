import json
from typing import Dict, List, Optional, Tuple, Union
from urllib.request import Request, urlopen

import streamlit as st


OVAREPRED_API_URL = "http://121.43.113.123:8888/api/portal/ovcalculate"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def interpolate_curve(x_value: float, anchors: List[Tuple[float, float]]) -> float:
    if x_value <= anchors[0][0]:
        return anchors[0][1]
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
        if x_value <= right_x:
            span = right_x - left_x
            if span == 0:
                return right_y
            ratio = (x_value - left_x) / span
            return left_y + ratio * (right_y - left_y)
    return anchors[-1][1]


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_ovarepred_reference(
    age: int,
    amh: float,
    fsh: float,
    afc: int,
    cycle_length: int,
    menarche_age: int,
) -> Optional[Dict[str, Union[float, str]]]:
    payload = {
        "age": int(age),
        "amh": float(amh),
        "fsh": float(fsh),
        "afc": int(afc),
        "startDay": int(cycle_length),
        "endDay": int(cycle_length),
        "menarcheAge": int(menarche_age),
        "calType": "aafamodel",
    }
    request = Request(
        OVAREPRED_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json;charset=UTF-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=3) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if raw.get("code") != 200 or not isinstance(raw.get("data"), dict):
        return None

    data = raw["data"]
    try:
        return {
            "source": "OvaRePred",
            "mode": "reference",
            "dor_age": float(data["dorBeginAge"]),
            "dor_age_lower": float(data["dorBeginAgeLower"]),
            "dor_age_upper": float(data["dorBeginAgeUpper"]),
            "peri_age": float(data["perimenopauseAge"]),
            "peri_age_lower": float(data["perimenopauseAgeLower"]),
            "peri_age_upper": float(data["perimenopauseAgeUpper"]),
            "tool_score": float(data["v1"]),
            "tool_level_cn": str(data.get("levelch") or ""),
            "tool_level_en": str(data.get("levelen") or ""),
        }
    except Exception:
        return None


def estimate_reserve_reference(age: int, reserve_score: float) -> Dict[str, Union[float, str]]:
    score = clamp(float(reserve_score), 0.0, 100.0)
    dor_gap = clamp((score - 30.0) * 0.22, 0.0, 18.0)
    peri_gap = clamp(dor_gap + 11.0, 8.0, 28.0)
    interval = clamp(3.6 - score * 0.02, 1.4, 3.6)
    dor_age = clamp(float(age) + dor_gap, float(age), 52.0)
    peri_age = clamp(float(age) + peri_gap, dor_age + 5.0, 60.0)
    return {
        "source": "Local estimate",
        "mode": "estimate",
        "dor_age": dor_age,
        "dor_age_lower": max(float(age), dor_age - interval),
        "dor_age_upper": dor_age + interval,
        "peri_age": peri_age,
        "peri_age_lower": max(dor_age + 3.5, peri_age - interval - 0.8),
        "peri_age_upper": peri_age + interval + 0.8,
        "tool_score": score,
        "tool_level_cn": "",
        "tool_level_en": "",
    }


def age_based_embryo_aneuploidy(age: float) -> Dict[str, Union[float, str]]:
    risk = interpolate_curve(
        float(age),
        [
            (25.0, 28.0),
            (30.0, 32.0),
            (35.0, 40.0),
            (37.0, 48.0),
            (40.0, 62.0),
            (42.0, 72.0),
            (45.0, 84.0),
            (50.0, 92.0),
        ],
    )
    risk = clamp(risk, 8.0, 96.0)
    euploid = 100.0 - risk
    if risk < 35:
        label = "相对较低"
    elif risk < 50:
        label = "逐步升高"
    elif risk < 70:
        label = "明显升高"
    else:
        label = "高风险区间"
    return {"risk": risk, "euploid": euploid, "label": label}
