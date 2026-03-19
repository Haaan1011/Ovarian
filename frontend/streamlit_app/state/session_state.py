import streamlit as st


def build_session_defaults(form_input_keys):
    defaults = {
        "cds": None,
        "model_ready": False,
        "model_source": "未加载",
        "model_error": None,
        "reserve_report": None,
        "reserve_patient": None,
        "reserve_error": None,
        "plan_report": None,
        "plan_patient": None,
        "plan_error": None,
        "uploaded_patients": [],
        "uploaded_file_digest": "",
        "uploaded_file_name": "",
        "current_patient_index": 0,
        "upload_notice": None,
    }
    defaults.update({key: "" for key in form_input_keys.values()})
    return defaults


def initialize_session_state(form_input_keys) -> None:
    for key, value in build_session_defaults(form_input_keys).items():
        if key not in st.session_state:
            st.session_state[key] = value
