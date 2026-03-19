from ovarian_prediction.clinical_system import ClinicalDecisionSystem
from ovarian_prediction.predict import OvarianPredictor
from ovarian_prediction.preprocessing import OvarianPreprocessor, make_synthetic_dataset
from ovarian_prediction.models import OvarianMLSystem


def test_pipeline_e2e():
    df = make_synthetic_dataset(n=300, random_state=42)
    proc = OvarianPreprocessor(random_state=42)
    data = proc.fit_transform(df)

    required_keys = {
        "porsm_train",
        "porsm_test",
        "horsm_train",
        "horsm_test",
        "pordm_train",
        "pordm_test",
        "hordm_train",
        "hordm_test",
    }
    assert required_keys.issubset(data)

    system = OvarianMLSystem(n_trials=2, random_state=42)
    system.train_all(data, tune=False)
    results = system.evaluate_all(data)
    assert set(results) == {"PORDM", "HORDM", "PORSM", "HORSM"}

    predictor = OvarianPredictor(system)
    probs = predictor.predict(
        {
            "AMH": 0.5,
            "AFC": 4,
            "FSH": 14.0,
            "LH": 6.5,
            "Age": 38,
            "P": 0.4,
            "Weight": 52,
            "DBP": 78,
            "WBC": 5.8,
            "ALT": 22,
            "RBC": 3.9,
            "Duration": 3,
            "POIorDOR": "Yes",
            "PCOS": "No",
            "PLT": 195,
            "Protocol": "Antagonist",
            "Initial.FSH": 225,
            "Recombinant": "Yes",
            "Use.LH": "Yes",
        }
    )
    assert set(probs) == {"prob_POR_dm", "prob_HOR_dm", "prob_POR_sm", "prob_HOR_sm"}

    cds = ClinicalDecisionSystem.from_ml_system(system)
    report = cds.evaluate_patient(
        {
            "AMH": 0.5,
            "AFC": 4,
            "FSH": 14.0,
            "LH": 6.5,
            "Age": 38,
            "P": 0.4,
            "Weight": 52,
            "DBP": 78,
            "WBC": 5.8,
            "ALT": 22,
            "RBC": 3.9,
            "Duration": 3,
            "POIorDOR": "Yes",
            "PCOS": "No",
            "PLT": 195,
        },
        {
            "Protocol": "Antagonist",
            "Initial.FSH": 225,
            "Recombinant": "Yes",
            "Use.LH": "Yes",
        },
    )
    assert "summary" in report
    assert report["risk_levels"]["POR"]["level"] in ["低", "中", "高"]
    assert report["risk_levels"]["HOR"]["level"] in ["低", "中", "高"]
