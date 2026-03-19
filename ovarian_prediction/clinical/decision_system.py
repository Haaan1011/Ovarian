from typing import Any, Dict, List, Optional

from ovarian_prediction.inference import OvarianPredictor
from ovarian_prediction.models import OvarianMLSystem

from .reporting import format_clinical_report
from .rules import (
    estimate_oocyte_count,
    risk_level,
    suggest_fsh_dose,
    suggest_fsh_type,
    suggest_lh,
    suggest_protocol,
)
from .thresholds import HOR_THRESHOLDS, POR_THRESHOLDS


class ClinicalDecisionSystem:
    """Clinical-facing facade that turns model outputs into recommendations."""

    def __init__(self, predictor: OvarianPredictor):
        self.predictor = predictor

    @classmethod
    def from_directory(cls, directory: str) -> "ClinicalDecisionSystem":
        return cls(OvarianPredictor.from_directory(directory))

    @classmethod
    def from_ml_system(cls, system: OvarianMLSystem) -> "ClinicalDecisionSystem":
        return cls(OvarianPredictor(system))

    def evaluate_patient(
        self,
        patient: Dict[str, Any],
        intervention: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        full_patient = {**patient}
        if intervention:
            full_patient.update(intervention)

        probs = self.predictor.predict(full_patient)
        por_prob = probs.get("prob_POR_sm", probs.get("prob_POR_dm", 0.0))
        hor_prob = probs.get("prob_HOR_sm", probs.get("prob_HOR_dm", 0.0))
        por_prob_dm = probs.get("prob_POR_dm")
        hor_prob_dm = probs.get("prob_HOR_dm")

        por_level = risk_level(por_prob, POR_THRESHOLDS)
        hor_level = risk_level(hor_prob, HOR_THRESHOLDS)

        recommendations = {
            "protocol": suggest_protocol(
                patient.get("AMH"),
                patient.get("AFC"),
                patient.get("FSH"),
                por_prob,
                hor_prob,
            ),
            "fsh_dose": suggest_fsh_dose(por_prob, hor_prob, patient.get("Weight")),
            "fsh_type": suggest_fsh_type(por_prob),
            "lh_support": suggest_lh(por_prob, patient.get("Age")),
            "est_oocytes": estimate_oocyte_count(
                patient.get("AMH"),
                patient.get("AFC"),
                patient.get("Age"),
            ),
        }

        return {
            "raw_probs": probs,
            "risk_levels": {
                "POR": {"prob": por_prob, "level": por_level},
                "HOR": {"prob": hor_prob, "level": hor_level},
            },
            "recommendations": recommendations,
            "summary": format_clinical_report(
                patient,
                probs,
                por_prob,
                hor_prob,
                por_level,
                hor_level,
                recommendations,
                por_prob_dm,
                hor_prob_dm,
            ),
        }

    def compare_interventions(
        self,
        patient: Dict[str, Any],
        interventions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        for intervention in interventions:
            result = self.evaluate_patient(patient, intervention)
            result["intervention"] = intervention
            results.append(result)
        return results
