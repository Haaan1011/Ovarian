from typing import Optional

from ovarian_prediction.models import OvarianMLSystem
from ovarian_prediction.preprocessing import OvarianPreprocessor, make_synthetic_dataset

from .decision_system import ClinicalDecisionSystem


def run_demo(ml_system: Optional[OvarianMLSystem] = None) -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║           卵巢反应预测系统 - 演示模式               ║")
    print("╚══════════════════════════════════════════════════════╝")

    if ml_system is None:
        print("\n[INFO] 未检测到预训练模型，使用合成数据快速训练演示模型...")
        df = make_synthetic_dataset(n=500, random_state=777)
        proc = OvarianPreprocessor()
        data = proc.fit_transform(df)
        ml_system = OvarianMLSystem(n_trials=10)
        ml_system.train_all(data, tune=False)

    cds = ClinicalDecisionSystem.from_ml_system(ml_system)

    patient_por = {
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
    }
    intervention_suggested = {
        "Protocol": "Antagonist",
        "Initial.FSH": 225,
        "Recombinant": "Yes",
        "Use.LH": "Yes",
    }
    print("\n" + "=" * 58)
    print("  📋 典型患者示例 1: 疑似DOR/卵巢低反应患者")
    print("=" * 58)
    print(cds.evaluate_patient(patient_por, intervention_suggested)["summary"])

    patient_hor = {
        "AMH": 6.8,
        "AFC": 22,
        "FSH": 5.1,
        "LH": 4.8,
        "Age": 27,
        "P": 0.3,
        "Weight": 55,
        "DBP": 72,
        "WBC": 6.2,
        "ALT": 18,
        "RBC": 4.3,
        "Duration": 1,
        "POIorDOR": "No",
        "PCOS": "Yes",
        "PLT": 260,
    }
    intervention_hos = {
        "Protocol": "Antagonist",
        "Initial.FSH": 75,
        "Recombinant": "Yes",
        "Use.LH": "No",
    }
    print("\n" + "=" * 58)
    print("  📋 典型患者示例 2: PCOS/疑似高反应患者")
    print("=" * 58)
    print(cds.evaluate_patient(patient_hor, intervention_hos)["summary"])
