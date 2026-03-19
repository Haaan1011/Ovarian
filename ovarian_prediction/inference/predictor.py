from typing import Any, Dict

from ovarian_prediction.models import OvarianMLSystem
from ovarian_prediction.preprocessing import HORRM_FEATURES, OS_INTERVENTIONS, PORRM_FEATURES

from .patient_adapter import patient_dict_to_df


class OvarianPredictor:
    """Inference facade for the four clinical prediction submodels."""

    def __init__(self, ml_system: OvarianMLSystem):
        self.system = ml_system

    def predict(self, patient: Dict[str, Any]) -> Dict[str, float]:
        results: Dict[str, float] = {}

        model_pordm = self.system.models["PORDM"]
        if model_pordm.model_ is not None:
            X_pordm = patient_dict_to_df(patient, features=PORRM_FEATURES, cat_cols=["POIorDOR"])
            results["prob_POR_dm"] = float(model_pordm.model_.predict_proba(model_pordm._align_features(X_pordm))[0, 1])

        model_hordm = self.system.models["HORDM"]
        if model_hordm.model_ is not None:
            X_hordm = patient_dict_to_df(
                patient,
                features=HORRM_FEATURES,
                cat_cols=["POIorDOR", "PCOS"],
            )
            results["prob_HOR_dm"] = float(model_hordm.model_.predict_proba(model_hordm._align_features(X_hordm))[0, 1])

        model_porsm = self.system.models["PORSM"]
        if model_porsm.model_ is not None:
            X_porsm = patient_dict_to_df(
                patient,
                features=PORRM_FEATURES + OS_INTERVENTIONS,
                cat_cols=["POIorDOR", "Protocol", "Recombinant", "Use.LH"],
            )
            results["prob_POR_sm"] = float(model_porsm.model_.predict_proba(model_porsm._align_features(X_porsm))[0, 1])

        model_horsm = self.system.models["HORSM"]
        if model_horsm.model_ is not None:
            X_horsm = patient_dict_to_df(
                patient,
                features=HORRM_FEATURES + OS_INTERVENTIONS,
                cat_cols=["POIorDOR", "PCOS", "Protocol", "Recombinant", "Use.LH"],
            )
            results["prob_HOR_sm"] = float(model_horsm.model_.predict_proba(model_horsm._align_features(X_horsm))[0, 1])

        return results

    @classmethod
    def from_directory(cls, directory: str) -> "OvarianPredictor":
        return cls(OvarianMLSystem.load(directory))
