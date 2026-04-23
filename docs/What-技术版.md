# Han_Ovarian 技术深挖说明

## 1. 系统架构总览

项目由四层组成：

1. 数据预处理层：清洗、切分、插补、编码；
2. 建模训练层：四个 XGBoost 子模型训练与评估；
3. 推理决策层：单患者概率推理 + 临床规则生成建议；
4. 交互展示层：Streamlit 双面板可视化与批量上传。

核心代码路径：

- 预处理：`ovarian_prediction/preprocessing/`
- 模型：`ovarian_prediction/models/`
- 推理：`ovarian_prediction/inference/`
- 临床规则：`ovarian_prediction/clinical/`
- 训练入口：`ovarian_prediction/training/cli.py`
- 前端入口：`frontend/streamlit_app/main.py`

## 2. 四模型体系设计

系统包含 4 个二分类子模型：

- `PORDM`：POR 诊断模型（不含干预特征）
- `HORDM`：HOR 诊断模型（不含干预特征）
- `PORSM`：POR 策略模型（包含干预特征）
- `HORSM`：HOR 策略模型（包含干预特征）

其中 DM（Diagnostic Model）强调患者基础状态，SM（Strategy Model）强调在干预条件下的风险变化。临床决策默认优先使用 SM 概率，若 SM 不可用则回退 DM 概率。

## 3. 预处理链路

`OvarianPreprocessor` 的关键步骤：

1. 删除无关字段；
2. 选择 POR/HOR 对应特征子集；
3. 按目标标签做分层训练/测试切分；
4. 使用 MICE 思路补全缺失值（`IterativeImputer + RandomForestRegressor`）；
5. 分类变量 one-hot 编码；
6. 对 SM 数据去除干预虚拟变量，得到对应 DM 数据。

输出为 8 个数据集：

- `porsm_train/test`
- `horsm_train/test`
- `pordm_train/test`
- `hordm_train/test`

## 4. 模型训练与评估

单模型由 `XGBSubmodel` 管理，支持：

- Optuna TPE 自动调参（AUC 最大化）；
- 固定参数快速训练；
- 特征对齐后推理；
- AUC 与 Brier 指标评估；
- 原生 XGBoost 格式保存（`*.json` + `*.meta.json`）。

四模型由 `OvarianMLSystem` 统一调度，支持批量训练、评估、保存、加载。

## 5. 推理与临床规则引擎

`OvarianPredictor`：

- 接收单患者字典；
- 按模型特征需求构造 DataFrame；
- 对齐特征后输出四个概率：
  - `prob_POR_dm`
  - `prob_HOR_dm`
  - `prob_POR_sm`
  - `prob_HOR_sm`

`ClinicalDecisionSystem`：

- 将概率映射为 POR/HOR 风险等级（低/中/高）；
- 规则函数输出：
  - 促排方案建议
  - FSH 剂量范围（支持体重微调）
  - FSH 类型建议
  - LH 支持建议
  - 预估获卵数区间
- 组装为结构化结果与临床摘要文本。

## 6. 前端系统实现要点

前端基于 Streamlit，关键特性：

- 左右双业务面板（储备评估、促排方案）；
- 输入校验和错误提示；
- Excel/CSV 批量上传与别名列头识别；
- 多患者选择与状态同步；
- 模型加载策略：
  1. 优先 `artifacts/models/xgboost/`
  2. 兼容 `models/`
  3. 若都不存在则自动训练演示模型回退
- 储备评估支持外部参考接口，失败时本地估算回退。

## 7. 模型产物与兼容策略

当前主产物目录：`artifacts/models/xgboost/`  
兼容旧产物目录：`models/`

兼容策略：

- 优先读取新格式 `json + meta.json`；
- 若仅有旧 `pkl`，可加载并迁移保存为新格式；
- 前端自动识别可用模型路径，减少部署摩擦。

## 8. 测试覆盖现状

测试目录 `tests/` 已覆盖：

- 单元测试：
  - 预处理
  - 模型训练推理
  - 临床规则
  - 上传解析
- 集成测试：
  - 合成数据 -> 预处理 -> 训练 -> 推理 -> 临床输出 端到端链路

说明当前工程并非仅“可运行”，而是具备基础可回归能力。

## 9. 工程阶段判断

项目处于“可演示可迭代”的工程化中期：

- 已完成核心功能闭环；
- 已有模型与测试基础；
- 下一阶段重点应是临床验证、阈值校准、部署运维化与可追溯治理。
