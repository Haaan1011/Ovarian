from ovarian_prediction.inference import OvarianPredictor
from ovarian_prediction.models import OvarianMLSystem
from ovarian_prediction.preprocessing import OvarianPreprocessor, make_synthetic_dataset


def test_predictor_returns_four_probabilities_after_quick_training():
    df = make_synthetic_dataset(n=120, random_state=42)
    proc = OvarianPreprocessor(random_state=42)
    data = proc.fit_transform(df)
    system = OvarianMLSystem(n_trials=1, random_state=42)
    system.train_all(data, tune=False)

    predictor = OvarianPredictor(system)
    result = predictor.predict(
        {
            "AMH": 2.0,
            "AFC": 11,
            "FSH": 7.5,
            "LH": 5.0,
            "Age": 31,
            "P": 0.4,
            "Weight": 55,
            "DBP": 75,
            "WBC": 6.0,
            "ALT": 22,
            "RBC": 4.2,
            "Duration": 2,
            "POIorDOR": "No",
            "PCOS": "No",
            "PLT": 230,
            "Protocol": "Long",
            "Initial.FSH": 150,
            "Recombinant": "Yes",
            "Use.LH": "No",
        }
    )
    assert set(result) == {"prob_POR_dm", "prob_HOR_dm", "prob_POR_sm", "prob_HOR_sm"}
