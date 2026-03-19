import joblib
import pandas as pd
import pytest

from frontend.streamlit_app.services import system_loader
from ovarian_prediction.models import XGBSubmodel


def test_xgb_submodel_trains_and_predicts_probabilities():
    train_df = pd.DataFrame(
        {
            "AMH": [0.8, 1.2, 2.1, 3.0, 3.4, 4.2],
            "AFC": [4, 5, 8, 12, 15, 18],
            "POR": ["Yes", "Yes", "No", "No", "No", "No"],
        }
    )
    model = XGBSubmodel("PORDM", "POR", n_trials=1, cv_folds=2, random_state=42)
    model.fit(train_df, params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1})
    probs = model.predict_proba(train_df.drop(columns=["POR"]))
    assert len(probs) == len(train_df)
    assert all(0.0 <= prob <= 1.0 for prob in probs)


def test_xgb_submodel_native_save_and_load_roundtrip(tmp_path):
    train_df = pd.DataFrame(
        {
            "AMH": [0.8, 1.2, 2.1, 3.0, 3.4, 4.2],
            "AFC": [4, 5, 8, 12, 15, 18],
            "POR": ["Yes", "Yes", "No", "No", "No", "No"],
        }
    )
    model = XGBSubmodel("PORDM", "POR", n_trials=1, cv_folds=2, random_state=42)
    model.fit(train_df, params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1})

    model.save(tmp_path)
    paths = XGBSubmodel.artifact_paths(tmp_path, "PORDM")

    assert paths["model"].exists()
    assert paths["metadata"].exists()

    reloaded = XGBSubmodel.load(str(paths["model"]))
    original_probs = model.predict_proba(train_df.drop(columns=["POR"]))
    reloaded_probs = reloaded.predict_proba(train_df.drop(columns=["POR"]))
    assert reloaded_probs.tolist() == pytest.approx(original_probs.tolist())


def test_xgb_submodel_loads_legacy_pickle_and_can_export_native_artifacts(tmp_path):
    train_df = pd.DataFrame(
        {
            "AMH": [0.8, 1.2, 2.1, 3.0, 3.4, 4.2],
            "AFC": [4, 5, 8, 12, 15, 18],
            "POR": ["Yes", "Yes", "No", "No", "No", "No"],
        }
    )
    model = XGBSubmodel("PORDM", "POR", n_trials=1, cv_folds=2, random_state=42)
    model.fit(train_df, params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1})

    legacy_path = tmp_path / "PORDM.pkl"
    joblib.dump(model, legacy_path)

    loaded = XGBSubmodel.load(str(legacy_path))
    loaded.save(tmp_path)
    paths = XGBSubmodel.artifact_paths(tmp_path, "PORDM")

    assert paths["model"].exists()
    assert paths["metadata"].exists()
    assert loaded.predict_proba(train_df.drop(columns=["POR"])).tolist() == pytest.approx(
        model.predict_proba(train_df.drop(columns=["POR"])).tolist()
    )


def test_resolve_model_directory_accepts_native_artifacts(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    legacy_dir = tmp_path / "models"
    artifact_dir.mkdir()
    legacy_dir.mkdir()
    (artifact_dir / "PORDM.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "PORDM.meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(system_loader, "MODEL_ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(system_loader, "LEGACY_MODEL_DIR", legacy_dir)

    assert system_loader.resolve_model_directory() == artifact_dir
