"""
test_pipeline.py
================
端到端测试: 合成数据 → 预处理 → 训练 → 预测 → 临床报告
"""

import sys
import os
# __file__ is ovarian_prediction/tests/test_pipeline.py
# 向上跳3级到 PredictOvarianResponse-main 目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import traceback


def test_preprocessing():
    print("\n" + "="*50)
    print("测试 1: 数据预处理模块")
    print("="*50)
    from ovarian_prediction.preprocessing import (
        make_synthetic_dataset, OvarianPreprocessor,
        MICEImputer, encode_categoricals
    )

    # 生成合成数据
    df = make_synthetic_dataset(n=300, random_state=42)
    print(f"  合成数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  POR 阳性率: {(df['POR']=='Yes').mean():.1%}")
    print(f"  HOR 阳性率: {(df['HOR']=='Yes').mean():.1%}")
    print(f"  AMH 缺失率: {df['AMH'].isna().mean():.1%}")

    # 预处理
    proc = OvarianPreprocessor(random_state=42)
    data = proc.fit_transform(df)

    required_keys = ["porsm_train", "porsm_test", "horsm_train", "horsm_test",
                     "pordm_train", "pordm_test", "hordm_train", "hordm_test"]
    for key in required_keys:
        assert key in data, f"缺少数据集: {key}"
        assert len(data[key]) > 0, f"数据集为空: {key}"
        assert "POR" in data[key].columns or "HOR" in data[key].columns, \
            f"数据集 {key} 缺少目标变量"

    print(f"  PORSM 训练集: {data['porsm_train'].shape}")
    print(f"  PORDM 测试集: {data['pordm_test'].shape}")
    print("  ✅ 预处理测试通过")
    return data


def test_model_training(data):
    print("\n" + "="*50)
    print("测试 2: 模型训练 (快速, 不调参)")
    print("="*50)
    from ovarian_prediction.models import OvarianMLSystem

    system = OvarianMLSystem(n_trials=5, random_state=42)
    system.train_all(data, tune=False)  # 快速训练

    # 验证模型都已训练
    for name, model in system.models.items():
        assert model.model_ is not None, f"{name} 未训练"
        assert model.feature_names_ is not None, f"{name} 缺少特征名"
        print(f"  {name}: 特征数={len(model.feature_names_)}")

    print("  ✅ 模型训练测试通过")
    return system


def test_evaluation(system, data):
    print("\n" + "="*50)
    print("测试 3: 模型评估")
    print("="*50)

    results = system.evaluate_all(data)
    for name, metrics in results.items():
        auc = metrics["auc"]
        bs = metrics["brier_score"]
        print(f"  {name}: AUC={auc:.4f}  Brier={bs:.4f}")
        assert 0.0 <= auc <= 1.0, f"{name} AUC 超出范围: {auc}"
        assert 0.0 <= bs <= 1.0, f"{name} Brier 超出范围: {bs}"

    print("  ✅ 模型评估测试通过")


def test_single_prediction(system):
    print("\n" + "="*50)
    print("测试 4: 单患者预测接口")
    print("="*50)
    from ovarian_prediction.predict import OvarianPredictor

    predictor = OvarianPredictor(system)

    # 典型 DOR 患者
    patient_por = {
        "AMH": 0.5, "AFC": 4, "FSH": 14.0, "LH": 6.5,
        "Age": 38, "P": 0.4, "Weight": 52, "DBP": 78,
        "WBC": 5.8, "ALT": 22, "RBC": 3.9, "Duration": 3,
        "POIorDOR": "Yes", "PCOS": "No", "PLT": 195,
        "Protocol": "Antagonist", "Initial.FSH": 225,
        "Recombinant": "Yes", "Use.LH": "Yes",
    }
    probs = predictor.predict(patient_por)
    print(f"  DOR患者预测结果: {probs}")
    for k, v in probs.items():
        assert 0.0 <= v <= 1.0, f"概率超出范围: {k}={v}"

    # 典型 PCOS 患者
    patient_hor = {
        "AMH": 6.8, "AFC": 22, "FSH": 5.1, "LH": 4.8,
        "Age": 27, "P": 0.3, "Weight": 55, "DBP": 72,
        "WBC": 6.2, "ALT": 18, "RBC": 4.3, "Duration": 1,
        "POIorDOR": "No", "PCOS": "Yes", "PLT": 260,
        "Protocol": "Antagonist", "Initial.FSH": 75,
        "Recombinant": "Yes", "Use.LH": "No",
    }
    probs2 = predictor.predict(patient_hor)
    print(f"  PCOS患者预测结果: {probs2}")

    # 测试缺失 AMH 的情况 (XGBoost 天然支持)
    patient_no_amh = {**patient_por, "AMH": None}
    probs3 = predictor.predict(patient_no_amh)
    print(f"  缺失AMH患者预测: {probs3}")

    print("  ✅ 单患者预测测试通过")


def test_clinical_system(system):
    print("\n" + "="*50)
    print("测试 5: 临床决策系统输出")
    print("="*50)
    from ovarian_prediction.clinical_system import ClinicalDecisionSystem

    cds = ClinicalDecisionSystem.from_ml_system(system)

    patient = {
        "AMH": 0.5, "AFC": 4, "FSH": 14.0, "LH": 6.5,
        "Age": 38, "P": 0.4, "Weight": 52, "DBP": 78,
        "WBC": 5.8, "ALT": 22, "RBC": 3.9, "Duration": 3,
        "POIorDOR": "Yes", "PCOS": "No", "PLT": 195,
    }
    intervention = {
        "Protocol": "Antagonist", "Initial.FSH": 225,
        "Recombinant": "Yes", "Use.LH": "Yes",
    }
    result = cds.evaluate_patient(patient, intervention)

    # 验证输出结构
    assert "raw_probs" in result
    assert "risk_levels" in result
    assert "recommendations" in result
    assert "summary" in result
    assert "POR" in result["risk_levels"]
    assert "HOR" in result["risk_levels"]
    assert result["risk_levels"]["POR"]["level"] in ["低", "中", "高"]
    assert result["risk_levels"]["HOR"]["level"] in ["低", "中", "高"]

    recs = result["recommendations"]
    required_recs = ["protocol", "fsh_dose", "fsh_type", "lh_support"]
    for key in required_recs:
        assert key in recs, f"建议缺少字段: {key}"

    print(f"\n{result['summary']}")
    print("  ✅ 临床系统测试通过")

    # 方案比较
    interventions = [
        {"Protocol": "Long",      "Initial.FSH": 150, "Recombinant": "Yes", "Use.LH": "No"},
        {"Protocol": "Antagonist","Initial.FSH": 225, "Recombinant": "Yes", "Use.LH": "Yes"},
    ]
    comparisons = cds.compare_interventions(patient, interventions)
    assert len(comparisons) == 2
    print(f"  方案对比: {len(comparisons)} 个方案已分析")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     卵巢反应预测系统 - 端到端测试                   ║")
    print("╚══════════════════════════════════════════════════════╝")

    all_passed = True
    try:
        data = test_preprocessing()
        system = test_model_training(data)
        test_evaluation(system, data)
        test_single_prediction(system)
        test_clinical_system(system)

        print("\n" + "="*50)
        print("🎉 所有测试通过!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
