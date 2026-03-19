import streamlit as st

from ovarian_prediction.clinical import ClinicalDecisionSystem
from ovarian_prediction.config import LEGACY_MODEL_DIR, MODEL_ARTIFACT_DIR
from ovarian_prediction.preprocessing import OvarianPreprocessor, make_synthetic_dataset
from ovarian_prediction.models import OvarianMLSystem


def _has_saved_models(candidate):
    native_model = candidate / "PORDM.json"
    native_metadata = candidate / "PORDM.meta.json"
    legacy_model = candidate / "PORDM.pkl"
    return (native_model.exists() and native_metadata.exists()) or legacy_model.exists()


def resolve_model_directory():
    for candidate in (MODEL_ARTIFACT_DIR, LEGACY_MODEL_DIR):
        if _has_saved_models(candidate):
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_system():
    model_directory = resolve_model_directory()
    if model_directory is not None:
        return ClinicalDecisionSystem.from_directory(str(model_directory)), "本地预训练模型"

    df = make_synthetic_dataset(n=200, random_state=42)
    proc = OvarianPreprocessor()
    data = proc.fit_transform(df)
    ml_system = OvarianMLSystem(n_trials=2)
    ml_system.train_all(data, tune=False)
    return ClinicalDecisionSystem.from_ml_system(ml_system), "演示模型"
