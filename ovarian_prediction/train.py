"""
train.py
========
命令行训练脚本

用法:
    # 使用真实数据训练
    python -m ovarian_prediction.train --data original_derivation.xlsx --output models/

    # 使用合成数据 (测试流程)
    python -m ovarian_prediction.train --synthetic --output models/ --no-tune
"""

import argparse
import os
import sys

from .preprocessing import make_synthetic_dataset, OvarianPreprocessor
from .models import OvarianMLSystem
from .clinical_system import run_demo


def main():
    parser = argparse.ArgumentParser(
        description="卵巢反应预测系统 - 模型训练")
    parser.add_argument("--data", type=str, default=None,
                        help="原始 Excel 数据文件路径 (e.g., original_derivation.xlsx)")
    parser.add_argument("--output", type=str, default="models",
                        help="模型输出目录 (默认: models/)")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据训练 (用于测试流程)")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="合成数据样本量 (默认: 1000)")
    parser.add_argument("--no-tune", action="store_true",
                        help="跳过超参数调优 (快速训练, 默认参数)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna 调参次数 (默认: 50)")
    parser.add_argument("--demo", action="store_true",
                        help="运行演示模式 (无需数据)")
    parser.add_argument("--seed", type=int, default=777, help="随机种子")
    args = parser.parse_args()

    if args.demo:
        print("启动演示模式...")
        run_demo()
        return

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    if args.synthetic or args.data is None:
        print(f"[INFO] 使用合成数据 (n={args.n_samples}) 训练模型...")
        df = make_synthetic_dataset(n=args.n_samples, random_state=args.seed)
    else:
        print(f"[INFO] 读取数据: {args.data}")
        import pandas as pd
        df = pd.read_excel(args.data)
        print(f"[INFO] 数据形状: {df.shape}")
        print(f"[INFO] 列名: {list(df.columns)}")

    # ── 预处理 ────────────────────────────────────────────────────────────────
    print("\n[STEP 1/3] 数据预处理 (MICE 填补 + 编码)...")
    proc = OvarianPreprocessor(random_state=args.seed)
    data = proc.fit_transform(df)

    print(f"  PORSM 训练集: {data['porsm_train'].shape}")
    print(f"  HORSM 训练集: {data['horsm_train'].shape}")

    # ── 训练模型 ──────────────────────────────────────────────────────────────
    tune = not args.no_tune
    print(f"\n[STEP 2/3] 训练4个子模型 (调参={'开启' if tune else '跳过'})...")
    ml_system = OvarianMLSystem(n_trials=args.n_trials, random_state=args.seed)
    ml_system.train_all(data, tune=tune)

    # ── 评估 ──────────────────────────────────────────────────────────────────
    print("\n[STEP 3/3] 模型评估...")
    results = ml_system.evaluate_all(data)
    print("\n【汇总指标】")
    for name, metrics in results.items():
        print(f"  {name}: AUC={metrics['auc']:.4f}  Brier={metrics['brier_score']:.4f}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    ml_system.save(args.output)
    proc.save(args.output)

    print(f"\n✅ 训练完成! 模型已保存至: {args.output}/")
    print("   运行演示: python -m ovarian_prediction.train --demo")


if __name__ == "__main__":
    main()
