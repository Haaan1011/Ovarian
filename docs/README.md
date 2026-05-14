# Predicting Ovarian Response

本仓是论文项目的独立工程：在 IVF/ICSI、GnRH-a 超长方案、首次促排治疗周期人群中，构建可解释 AI 辅助的个体化促排策略推荐与临床结局预测系统。

## 当前主线

- 唯一可写主线：`ssh zhishi-ts` 上的 `~/Han_PredictCP`
- 运行环境：`Han_Overian`
- GitHub：`git@github.com:Haaan1011/Predicting_Ovarian.git`
- 日常状态只看：`docs/status.md`、`docs/PLANS.md`

进入远端环境：

```bash
ssh zhishi-ts
source /home/zhishi/anaconda3/etc/profile.d/conda.sh
conda activate Han_Overian
cd ~/Han_PredictCP
```

## 正式 cohort

正式训练、评估、解释和 UI 默认使用以下三条同时满足的人群：

1. IVF/ICSI 相关字段
2. GnRH-a 超长方案
3. 首次促排治疗周期，即治疗次数 = 1

当前固定 flow：`13900 -> 12639 -> 6696 -> 4071`。

## 三层任务

- Layer 1：下一次 Gn 调整方向分类（加量 / 维持 / 减量）+ KNN 相似病例解释；具体剂量只作为候选策略评分辅助，不再作为主监督学习目标。
- Layer 2：获卵数、MII、OHSS 风险。
- Layer 3：临床妊娠、活产。

关键目标定义：

- `target_mii = 获卵数 - GV - MI`
- `target_live_birth = 早产 + 足月产 > 0`
- 第三层不接入胚胎/移植变量。

## 目录说明

- `data/`：数据层。`data/raw/hospital_excel/` 放本地原始 Excel，`data/interim/` 放标准化中间表，`data/processed/` 放建模样本。
- `preprocessing/`：数据 intake、清洗、标准化、字段映射。
- `features/`：特征工程、动态 forward fill、layer1/layer2/layer3 样本构建。
- `models/`：模型训练公共模块、模型注册、layer1/layer2/layer3 相关逻辑与模型产物。
- `evaluation/`：指标、校准、比较和评估报告模块。
- `explainability/`：SHAP 和病例解释模块。
- `prototype/streamlit_app/`：Streamlit 原型界面。
- `ui_design/`：UI 信息架构、组件规范、参考图和线框。
- `configs/`：cohort、schema、features、targets、models、UI 配置。
- `scripts/`：命令入口。训练、审计、样本构建、TensorBoard 生成都从这里执行。
- `docs/`：项目状态、计划、审计和阶段记录。日常优先看 `docs/status.md` 和 `docs/PLANS.md`。
- `skills/`：项目级 Codex skills。
- `tests/`：单元测试和回归检查。

## data 目录详细说明

`data/` 是本项目最重要的可审计目录。原则是：

- `data/raw/` 只保存原始输入，不作为训练直接读取。
- `data/interim/` 保存标准化后的中间层结果，用于核对字段语义、时序和缺失处理。
- `data/processed/` 保存可直接训练的样本层，必须能由脚本重复生成。
- `data/profiles/` 保存数据画像、审计、排除因素、字段覆盖等报告。
- `data/dictionary/` 保存原始字段到标准化字段的映射字典。
- `data/splits/` 保存统一切分文件，保证所有层共用同一套 train/valid/test 口径。

### 1. `data/raw/`

- `data/raw/hospital_excel/`：医院原始 Excel，包含临床资料表和监测表。这里是源数据，不做训练读取。
- `data/raw/.gitkeep`：保留空目录结构用，不是业务文件。

### 2. `data/dictionary/`

- `clinical_table_field_mapping.csv`：临床资料表原始字段到标准化字段的映射表。
- `monitoring_table_field_mapping.csv`：监测表原始字段到标准化字段的映射表。
- 这两个文件主要用于审计“原始字段到底被解释成了什么”，也是后续字段回溯的依据。

### 3. `data/interim/`

- `clinical_standardized.csv`：临床资料表标准化结果，保留中文原始语义与统一字段结构。
- `monitoring_visits_standardized.csv`：监测 visit 级标准化表，保留第几次监测、日期锚点和核心激素信息。
- `monitoring_medications_long.csv`：监测表中的药物明细长表，把重复药物字段展开成一行一药物的结构。
- `monitoring_follicles_long.csv`：监测表中的卵泡明细长表，把卵泡直径与数目展开成长表。
- `cycle_master_index.csv`：周期级主索引，连接临床表与监测表。
- `standardization_report.md`、`standardization_summary.json`：标准化过程报告和摘要，说明用了什么规则解决了什么问题。

### 4. `data/processed/`

- `baseline_cycle_dataset.csv`：周期级基线样本，主要用于 cohort / 基线分析。
- `snapshot_feature_dataset.csv`：快照特征母表，一行代表一个周期在一个监测时点的可见状态。
- `layer1_strategy_dataset.csv`：第一层策略训练样本，只保留能够构造下一次 Gn 动作标签的快照。
- `layer2_snapshot_dataset.csv`：第二层中间结局样本，用于获卵数、MII、OHSS 预测。
- `layer3_snapshot_dataset.csv`：第三层终点样本，用于临床妊娠、活产预测。
- `feature_manifest.json`：样本特征清单，说明每个 processed 数据集到底用了哪些列。
- `sample_build_report.md`、`sample_build_summary.json`：样本构建报告和摘要，说明样本量、过滤规则和排除项。

### 5. `data/profiles/`

- `ingest_manifest.json`：原始数据 intake 清单。
- `clinical_table_columns.csv`、`clinical_table_report.md`、`clinical_table_summary.json`：临床表画像与说明。
- `monitoring_table_columns.csv`、`monitoring_table_report.md`、`monitoring_table_summary.json`：监测表画像与说明。
- `monitoring_sheet_summary.csv`、`monitoring_union_column_coverage.csv`：监测各分表字段覆盖情况。
- `factor_mapping_audit_v2.csv`：4.22 因素映射审计结果。
- `raw_field_semantics_audit_v2.csv`：原始字段语义解释审计。
- `exclusion_factors_audit_v2.csv`：明确排除项和原因。
- `official_cohort_flow_v2.csv`、`official_cohort_data_audit_report_v2.md`、`official_cohort_data_audit_summary_v2.json`：正式 cohort flow 和审计结论。
- `kong_preprocessing_v1_missingness_audit.csv`、`kong_preprocessing_v1_registry_columns.csv`：缺失率和预处理注册表审计。
- `carry_forward_audit_v2.csv`：监测表行位继承/补值风险审计。

### 6. `data/splits/`

- `split_manifest_v1.csv`：统一切分表，所有层训练、验证、测试共用。
- `data/splits/.gitkeep`：保留目录结构。

### 7. 读法建议

- 先看 `data/profiles/official_cohort_flow_v2.csv`，再看 `data/interim/standardization_report.md`，最后看 `data/processed/sample_build_report.md`。
- 如果要核对某个字段怎么来的，优先查 `data/dictionary/*_field_mapping.csv`。
- 如果要看某个样本层到底用到了哪些列，优先查 `data/processed/feature_manifest.json`。

## 常用命令

### 数据 intake / 标准化

```bash
python scripts/ingest/run_ingest.py
python scripts/ingest/run_standardize.py
python scripts/ingest/run_male_factor_augment.py
```

### 数据审计

```bash
python scripts/audit/run_data_audit.py
```

### 样本构建

```bash
python scripts/build_samples/run_build_samples.py
```

### 模型训练

Layer1 的正式主目标已改为 Gn 调整方向三分类；Layer2 / Layer3 仍使用统一 `run_train.py`。


#### Layer1 Gn 调整方向分类 + KNN 相似病例解释

单目标 combined Gn action 训练：

```bash
python scripts/train/run_layer1_action_train.py --threshold 37.5 --target combined_gn_action --models lightgbm xgboost catboost --knn-k 50 --knn-report-limit 25
```

FSH / LH / HMG 三个拆分剂量 action 的正式优化训练：

```bash
python scripts/experiment/run_layer1_splitdose_action_optimization.py --threshold 75 --targets fsh_action lh_action hmg_action
```

训练完成后会自动写入 TensorBoard display：`models/tensorboard_display/run/<RUN_ID>/_overview`，其中包含最终指标表、类别分布表、SHAP 图、混淆矩阵、预测分布图、KNN selection/success 图和相似病例表。

当前拆分剂量正式候选 run 为 `phase8_layer1_splitdose_action_opt_thr75_20260512_192404`。FSH/LH/HMG 分别输出独立 bundle，并登记在 `models/artifacts/current_layer1_split_action_runs.json`。37.5 IU 对照已保留为审计参考，但拆分剂量正式候选暂采用 75 IU 阈值，因为 FSH 与 HMG 的验证/测试表现更稳定。

主要输出：

- `layer1_gn_action_metrics.csv`：Accuracy、Macro-F1、Weighted-F1、Precision、Recall、各类 support。
- `layer1_gn_action_confusion_matrix.png`：加量 / 维持 / 减量混淆矩阵。
- `layer1_gn_action_shap_top_features.png`：三分类模型 SHAP top features。
- `layer1_gn_action_predictions.csv`：每个测试 snapshot 的预测动作和三类概率。
- `layer1_knn_similar_action_stats.csv`：相似病例 action selection rate 和 success rate。
- `layer1_knn_similar_patient_table.csv`：K 个相似历史病例明细。
- `layer1_knn_selection_rate.png`、`layer1_knn_success_rate.png`、`similar_case_distance_plot.png`：KNN 解释图。

#### 全量训练

```bash
python scripts/train/run_train.py --experiment-config configs/models/experiment_v1.yaml
```

#### Layer2 获卵数 / MII

```bash
python scripts/train/run_train.py --experiment-config configs/models/experiment_v1.yaml --tasks layer2_oocytes layer2_mii
```

#### Layer2 OHSS

```bash
python scripts/train/run_train.py --experiment-config configs/models/experiment_v1.yaml --tasks layer2_ohss
```

#### Layer3 临床妊娠 / 活产

```bash
python scripts/train/run_train.py --experiment-config configs/models/experiment_v1.yaml --tasks layer3_live_birth layer3_clinical_pregnancy
```

#### 指定模型

```bash
python scripts/train/run_train.py --experiment-config configs/models/experiment_v1.yaml --tasks layer2_oocytes --models lightgbm catboost
```

#### 强制重建切分

```bash
python scripts/train/run_train.py --experiment-config configs/models/experiment_v1.yaml --rebuild-splits
```



### 模型质量诊断

```bash
python scripts/diagnose_phase5_quality.py --run-id <RUN_ID>
python scripts/diagnose_ohss_threshold_calibration.py --run-id <RUN_ID>
python scripts/diagnose_ohss_cycle_level.py --run-id <RUN_ID>
```

### TensorBoard 展示日志

TensorBoard 只保留训练过程曲线、SHAP 图片/文本和最终结果表格；不再把最终静态指标写成一条直线的 scalar 曲线。

```bash
python scripts/build_tensorboard_report.py --run-id <RUN_ID> --output-root models/tensorboard_display
```

启动 TensorBoard：

```bash
tensorboard --logdir models/tensorboard_display/run --host 127.0.0.1 --port 6006
```

### Layer1 action inference service

Streamlit 结果页优先读取最新且实际包含 `layer1_gn_action_best_bundle.joblib` 的 `models/artifacts/phase8_layer1_action_*/layer1_action/` 目录，并基于 `data/processed/layer1_strategy_dataset.csv` 的训练集历史库实时生成 KNN 相似病例证据。若 bundle 内包含 `decision_weights`，推理服务会先校准三类概率再输出推荐动作；若 bundle 或历史库不可用，页面会回退到前端示例逻辑。

### Streamlit UI

```bash
streamlit run prototype/streamlit_app/app.py --server.address 127.0.0.1 --server.port 8502
```

### 基础验证

```bash
python -m compileall common preprocessing features models evaluation explainability scripts prototype tests
pytest -q
```

## 当前模型结论摘要

- Layer2 获卵数与 MII 已明显优于 baseline，但仍需继续提升 RMSE/R2。
- OHSS 是当前最高风险任务，PR-AUC 和概率校准仍需重点优化。
- Layer3 当前采用 last1 + layer2 OOF stack 的稳定融合思路，临床妊娠和活产更适合作为预后倾向性参考。
- GPU/CPU 同参对照已完成：当前数据规模下不直接切换 GPU 作为默认训练，只把 GPU 用于后续大规模调参。


## Layer 1 KNN Similar-Patient Explanation

Layer 1 首先基于当前 snapshot 特征预测下一次 Gn 调整方向：`increase`、`maintain`、`decrease`。随后 KNN 模块只使用当前时点可见信息，在训练集/历史库中检索相似病例，统计医生历史选择率 selection rate，以及不同动作下的卵巢反应成功率、MII 成功率、OHSS-free 率、临床妊娠率和活产率。

KNN 匹配特征不包含 `next_gn_dose`、action 标签、获卵数、MII、OHSS、临床妊娠、活产、胚胎或移植变量；验证/测试样本检索时排除同一患者和同一周期。该模块用于解释和临床讨论，不代表因果推断或自动处方。

## Git 注意事项

- 原始 Excel、`data/interim`、`data/processed`、`models/artifacts`、TensorBoard 日志默认不提交。
- 提交前先检查：

```bash
git status -sb
git diff --stat
```
