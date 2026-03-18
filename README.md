# 🧬 IVF卵巢反应人工智能预测系统 (Python版)

本仓库是一个针对体外受精 (IVF) 临床专用的**卵巢反应预测与促排策略推荐辅助决策系统**。
最初系统由 R 语言开发（基于 `tidymodels` 和 `miceRanger`），本项目已将其**全部转化、重构并升级为了完整的 Python 模块化工程**，同时附带了极具科技感的高层级 Web 交互端。

---

## 🎯 系统核心功能与价值

本系统主要用于辅助生殖医生在进行试管婴儿促排卵前，通过患者的少量基础体征数据评估其卵巢反应风险，以达到**精准控药、规避风险、提高获卵率**的目的：

1. **评估卵巢低反应 (POR/DOR) 风险**：防止由于促排卵药物剂量不够导致的无卵可用。
2. **评估卵巢高反应 (OHSS/HOR) 风险**：防止卵巢过度刺激综合征这一严重并发症的发生。
3. **提供智能促排策略建议**：基于 XGBoost 策略模型，能够动态推演“如果采用某类方案、某剂量能降低哪些风险”，并直接输出用药建议反馈给医生。
4. **支持特征缺失 (MICE容错补全)**：即使患者某些化验单结果不全，系统内部构建的 随机森林多重插补器 也能推测并照常输出精准结果。

---

## 📂 项目模块详细说明 (代码是干什么的？)

该系统在 Python 层级做了高度的代码解耦，结构非常清晰。核心代码全部被放置在 `ovarian_prediction/` 包中。

### 1. 核心后端代码树
```text
PredictOvarianResponse-main/
│
├── ovarian_prediction/        ⬅️ 核心 AI 引擎包
│   ├── __init__.py            
│   ├── preprocessing.py       ✅ [数据工厂] 清洗多余数据、MICE 缺失值插补、特征 One-Hot 编码的管线
│   ├── models.py              ✅ [AI大脑] XGBoost 4个子分类器的定义，及 Optuna 贝叶斯最优超参数调优的实现
│   ├── predict.py             ✅ [推理机] 单患者数据的清洗验证，并对接给模型获取预测概率
│   ├── clinical_system.py     ✅ [临床大脑] 将冷冰冰的概率根据循证医学规则转化为具体的“医嘱文字”和“用药建议”
│   ├── train.py               ✅ [调度员] 命令行独立脚本，用于在服务器后台一键喂入Excel数据训练出新模型
│   └── requirements.txt       ✅ [依赖包] 系统运行需要什么外部包支持，全部写在这里
│
├── app.py                     ✅ [展现层] 基于 Streamlit 开发的动态 Web 大屏页面（极具高级科技感）
└── README.md                  ✅ [说明书] 本文档
```

### 2. 关于内部的 4 大 XGBoost 子模型

我们的系统并不是跑一个普通的模型了事，它底层拥有 **4个独立的 XGBoost 子模型** 并行工作相互校验：

* `PORDM`：卵巢低反应诊断模型（仅用患者基础体征，不引入干预方案）。
* `HORDM`：卵巢高反应诊断模型（同上）。
* `PORSM`：卵巢低反应 **策略(Strategy)** 模型（除了体征，还要输入该患者拟采用的促排方案，看此方案是否能降低低反应率）。
* `HORSM`：卵巢高反应 **策略(Strategy)** 模型（同上，评估此方案会不会激惹 OHSS）。

---

## 🚀 指令操作大全 (如何运行？)

无论您是数据分析师、Python开发还是临床部署工程师，请根据需求服用以下指令。

### 一、准备环境
打开该项目所在的 Terminal (终端)，第一步永远是安装外部 Python 依赖包：
```bash
cd /home/zhishi/PredictOvarianResponse-main
pip install -r ovarian_prediction/requirements.txt
```
> *注：用到的核心库包括 xgboost, scikit-learn, optuna, pandas 等。*

### 二、日常使用：一键拉起 Web 临床评估前台
如果您想直接通过可视化网页输入参数并拿给专家或患者看，不需要执行复杂的代码：
```bash
streamlit run app.py
```
终端会输出一个本地网络地址 `http://localhost:8501/`，用浏览器打开即可。在这个界面，你可以享受极具动态和动画高级感的卵巢反应预测平台。

### 三、后端开发：如何用您医院的真实数据从零训练这 4 个模型？
因为原始 R 语言论文没有公开附带真实的医院 EXCEL 数据表（存在隐私政策），如果您手里有收集到的本地 Excel 文件（比如名叫 `original_derivation.xlsx`），您可以通过下面这行指令一键建立新模型：
```bash
python -m ovarian_prediction.train --data original_derivation.xlsx --output models/
```
**这条指令做了什么？**
1. 读取传入的 Excel。
2. 内部切分 70% 训练 30% 测试集。
3. 对缺失值进行 MICE 随机森林迭代填补。
4. Optuna 循环 50 次使用贝叶斯探索寻找当下医院数据最完美的 XGBoost 树深和学习率。
5. 打印 AUC 评估并把成型的模型生成保存到 `models/` 文件夹下方。

**如果您只想快速跑通测试看看代码效果（跳过漫长调参，用系统自己生成的假患者数据试运行）**：
```bash
python -m ovarian_prediction.train --synthetic --no-tune
```

### 四、快速在命令行里输出"医嘱打印小条"演示
```bash
python -m ovarian_prediction.train --demo
```
终端会模拟两个典型患者（一个是多囊卵巢易激惹的患者，一个是卵巢早衰低反应患者），直接把基于其状态生成的风险率、推荐长方案还是拮抗剂、是否推荐补充 LH 等中文长文本打印在命令行里，非常适合用来检查您的决策树链路。

---

## 💡 给后续接手开发同事的避坑指南 (Q&A)

1. **模型预测时的 `LabelEncoder` 报错？**
   由于 xgboost 更新和 sklearn 的严格模式，在重构中已剔除了易导致未知字符串排序映射乱序的 LabelEncoder，内部 `models.py` 改用健壮的硬编码 `np.where(== "Yes")` 进行 0/1 判定，请勿再往系统里引入自动 Encoder。
   
2. **如果在输入前台把某个参数留空了会报错吗？**
   **不会！** 这是本系统的最大亮点。您在前台留下空白，传递给后台就会变成 `np.nan` 或 `None`。XGBoost 引擎天生具备对缺失叶子节点划分的容错支持，同时 MICE 填补器也会发挥作用进行特征代理。因此不必强求患者输入所有特征栏。
   
3. **想修改“中高低”风险的截断概率？**
   原本 R 语言文献有特定的 ROC 阈值设定。如今全被整合到了 `ovarian_prediction/clinical_system.py` 顶部常量 `POR_THRESHOLDS` (0.2/0.4) 和 `HOR_THRESHOLDS` (0.2/0.35)。要改随手即可改，无须重训 AI。
