import argparse
from pathlib import Path

import pandas as pd

from ovarian_prediction.clinical import run_demo
from ovarian_prediction.config import LEGACY_MODEL_DIR, MODEL_ARTIFACT_DIR, PREPROCESSOR_ARTIFACT_DIR
from ovarian_prediction.preprocessing import make_synthetic_dataset

from .orchestrator import train_from_dataframe


def main():
    parser = argparse.ArgumentParser(description="卵巢反应预测系统 - 模型训练")
    parser.add_argument("--data", type=str, default=None, help="原始 Excel 数据文件路径")
    parser.add_argument("--output", type=str, default=None, help="模型输出目录")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据训练")
    parser.add_argument("--n-samples", type=int, default=1000, help="合成数据样本量")
    parser.add_argument("--no-tune", action="store_true", help="跳过超参数调优")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna 调参次数")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--seed", type=int, default=777, help="随机种子")
    args = parser.parse_args()

    if args.demo:
        print("启动演示模式...")
        run_demo()
        return

    if args.synthetic or args.data is None:
        print(f"[INFO] 使用合成数据 (n={args.n_samples}) 训练模型...")
        df = make_synthetic_dataset(n=args.n_samples, random_state=args.seed)
    else:
        print(f"[INFO] 读取数据: {args.data}")
        df = pd.read_excel(args.data)
        print(f"[INFO] 数据形状: {df.shape}")
        print(f"[INFO] 列名: {list(df.columns)}")

    tune = not args.no_tune
    print("\n[STEP 1/3] 数据预处理 (MICE 填补 + 编码)...")
    proc, ml_system, results = train_from_dataframe(
        df,
        tune=tune,
        n_trials=args.n_trials,
        random_state=args.seed,
    )

    print(f"\n[STEP 2/3] 训练4个子模型 (调参={'开启' if tune else '跳过'})...")
    print("\n[STEP 3/3] 模型评估...")
    print("\n【汇总指标】")
    for name, metrics in results.items():
        print(f"  {name}: AUC={metrics['auc']:.4f}  Brier={metrics['brier_score']:.4f}")

    model_output = Path(args.output) if args.output else MODEL_ARTIFACT_DIR
    preprocessor_output = PREPROCESSOR_ARTIFACT_DIR if args.output is None else Path(args.output)
    legacy_output = LEGACY_MODEL_DIR if args.output is None else None

    model_output.mkdir(parents=True, exist_ok=True)
    preprocessor_output.mkdir(parents=True, exist_ok=True)
    ml_system.save(str(model_output))
    proc.save(str(preprocessor_output))
    if legacy_output is not None:
        legacy_output.mkdir(parents=True, exist_ok=True)
        ml_system.save(str(legacy_output))
        proc.save(str(legacy_output))

    print(f"\n✅ 训练完成! 模型已保存至: {model_output}/")
    print("   运行演示: python -m ovarian_prediction.train --demo")
