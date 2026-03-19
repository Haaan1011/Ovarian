from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
FRONTEND_DIR = REPO_ROOT / "frontend" / "streamlit_app"
ASSETS_DIR = FRONTEND_DIR / "assets"
BRAND_IMAGES_DIR = ASSETS_DIR / "images" / "brand"
BACKGROUND_IMAGES_DIR = ASSETS_DIR / "images" / "background"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_ARTIFACT_DIR = ARTIFACTS_DIR / "models" / "xgboost"
PREPROCESSOR_ARTIFACT_DIR = ARTIFACTS_DIR / "preprocessors" / "mice"
LEGACY_MODEL_DIR = REPO_ROOT / "models"
