# Han_Ovarian Project Overview

## 1. What this project does

Han_Ovarian is an AI-assisted clinical decision support project for IVF ovarian response assessment.  
Its goal is to estimate patient risk before stimulation and provide practical recommendations for ovarian stimulation planning.

Main objectives:

- Predict low ovarian response risk (POR);
- Predict high ovarian response / OHSS risk (HOR);
- Generate actionable stimulation recommendations (protocol direction, FSH starting dose, LH support, etc.);
- Provide an interactive clinical UI with batch patient input support.

## 2. Current project stage

Based on code structure, tests, and model artifacts, the project is at a **working end-to-end MVP+ stage**:

- Full training pipeline is implemented;
- Full inference + recommendation pipeline is implemented;
- Streamlit frontend is operational with single and batch workflows;
- Unit and integration tests are available;
- Trained model artifacts already exist in `artifacts/models/xgboost/`;
- Legacy compatibility for old model format/path is preserved.

In short, this is beyond a pure research prototype and is ready for iterative production-hardening.

## 3. Methods and technical approach

### 3.1 Modeling strategy

The system uses four parallel XGBoost binary classifiers:

- `PORDM` (POR diagnostic model)
- `HORDM` (HOR diagnostic model)
- `PORSM` (POR strategy model with intervention features)
- `HORSM` (HOR strategy model with intervention features)

Training supports:

- Fast fixed-parameter training;
- Optuna-based hyperparameter optimization (TPE sampler, AUC target).

### 3.2 Data preprocessing

- Stratified train/test split by target;
- Missing-value imputation with MICE-style iterative imputer
  (`IterativeImputer + RandomForestRegressor`);
- Categorical one-hot encoding;
- Feature set separation between diagnostic and strategy models.

### 3.3 Inference and clinical logic

- Patient dict is adapted to model-aligned feature vectors;
- Four probabilities are produced (`prob_POR_dm`, `prob_HOR_dm`, `prob_POR_sm`, `prob_HOR_sm`);
- Rule engine maps probabilities to low/medium/high risk levels;
- Recommendation functions generate:
  - protocol suggestion
  - FSH dose range
  - FSH type suggestion
  - LH support suggestion
  - estimated oocyte count
- Clinical summary text is generated for direct use.

### 3.4 Frontend interaction

- Streamlit dual-panel workflow:
  - ovarian reserve assessment
  - stimulation planning
- Excel/CSV upload, template download, alias-based header recognition;
- Multi-patient selection and state synchronization;
- Auto fallback strategy:
  1. load local pretrained artifacts
  2. fallback to legacy model directory
  3. fallback to on-the-fly demo model training.

## 4. Implemented features

### 4.1 Training and evaluation

- CLI-based training entry (`real data` / `synthetic data` / `demo mode`);
- Four-model batch train and evaluate;
- AUC and Brier score reporting;
- Artifact save/load with native JSON+metadata and legacy pickle compatibility.

### 4.2 Inference and decision support

- Single-patient four-risk probability output;
- Risk stratification and recommendation generation;
- Intervention comparison support via decision system API.

### 4.3 Frontend capabilities

- Input validation and error handling;
- Visualization cards and risk displays;
- Batch patient upload and switching;
- Model-source awareness (pretrained vs demo).

### 4.4 Engineering quality

- Unit tests for preprocessing, models, inference, clinical rules, upload parser;
- Integration test for full E2E pipeline.

## 5. Folder responsibility map

- `ovarian_prediction/`: core backend engine
  - `preprocessing/`, `models/`, `inference/`, `clinical/`, `training/`, `config/`
- `frontend/streamlit_app/`: Streamlit UI and interaction services
- `artifacts/`: generated model and runtime artifacts
- `models/`: legacy artifact compatibility path
- `docs/`: project and research documentation
- `tests/`: unit + integration test suites
- `scripts/`: reserved script directory

## 6. One-line summary

Han_Ovarian already has a complete train-to-clinical-output loop with usable UI and tests; the next major step is clinical validation, threshold calibration, and deployment-level robustness.
