import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import streamlit as st


UPLOAD_HEADER_ALIASES = {
    "patient_name": {"姓名", "患者姓名", "patientname", "name"},
    "patient_id": {"患者编号", "编号", "病历号", "id", "patientid", "caseid"},
    "menarche_age": {"初潮年龄", "月经年龄", "menarcheage", "menstrualage"},
    "cycle_length": {
        "月经周期",
        "月经周期天",
        "月经周期天数",
        "cyclelength",
        "menstrualcycle",
        "menstrualcycledays",
    },
    "BMI": {"bmi", "体重指数"},
    "AMH": {"amh", "amhngml", "antimullerianhormone"},
    "AFC": {"afc", "窦卵泡数", "窦卵泡计数", "antralfolliclecount"},
    "LH": {"lh", "basallh", "基础lh"},
    "FSH": {"fsh", "bfsh", "基础fsh", "basalfsh", "bfs"},
    "Age": {"年龄", "age", "年龄years", "womanage", "patientage"},
}


def normalize_header_name(text: str) -> str:
    value = str(text or "").strip().lower()
    value = (
        value.replace("（", "(")
        .replace("）", ")")
        .replace("／", "/")
        .replace("—", "-")
        .replace("–", "-")
    )
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value)


def canonical_upload_field(header: str) -> Optional[str]:
    normalized = normalize_header_name(header)
    if not normalized:
        return None
    for field, aliases in UPLOAD_HEADER_ALIASES.items():
        if normalized in aliases:
            return field
    for field, aliases in UPLOAD_HEADER_ALIASES.items():
        if any(alias and alias in normalized for alias in aliases):
            return field
    return None


def coerce_optional_number(value) -> Optional[Union[int, float]]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


@st.cache_data(show_spinner=False)
def build_upload_template_bytes() -> bytes:
    template = pd.DataFrame(
        [
            {
                "patient_id": "FJFY-001",
                "patient_name": "患者A",
                "Age": 32,
                "AMH": 2.5,
                "FSH": 8.0,
                "AFC": 10,
                "cycle_length": 28,
                "menarche_age": 13,
                "BMI": 21.3,
                "LH": 6.0,
            },
            {
                "patient_id": "FJFY-002",
                "patient_name": "患者B",
                "Age": 37,
                "AMH": 1.4,
                "FSH": 9.8,
                "AFC": 7,
                "cycle_length": 29,
                "menarche_age": 14,
                "BMI": 22.1,
                "LH": 5.4,
            },
        ]
    )
    buffer = BytesIO()
    template.to_excel(buffer, index=False)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_uploaded_patients(
    file_bytes: bytes,
    filename: str,
) -> List[Dict[str, Union[str, int, float, None]]]:
    suffix = Path(filename).suffix.lower()
    bio = BytesIO(file_bytes)
    frame = pd.read_csv(bio) if suffix == ".csv" else pd.read_excel(bio)

    columns = {}
    for column in frame.columns:
        field = canonical_upload_field(str(column))
        if field and field not in columns:
            columns[field] = column

    patients: List[Dict[str, Union[str, int, float, None]]] = []
    for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        patient = {
            "row_index": row_index,
            "patient_name": None,
            "patient_id": None,
            "Age": None,
            "AMH": None,
            "FSH": None,
            "AFC": None,
            "cycle_length": None,
            "menarche_age": None,
            "BMI": None,
            "LH": None,
        }
        for field, column in columns.items():
            raw_value = row[column]
            if field in {"patient_name", "patient_id"}:
                if raw_value is not None and not pd.isna(raw_value):
                    text = str(raw_value).strip()
                    patient[field] = text or None
            else:
                patient[field] = coerce_optional_number(raw_value)
        if any(
            patient.get(key) is not None
            for key in ["Age", "AMH", "FSH", "AFC", "cycle_length", "menarche_age", "BMI", "LH"]
        ):
            label = patient.get("patient_name") or patient.get("patient_id") or f"患者 {row_index}"
            patient["patient_label"] = label
            patients.append(patient)
    return patients
