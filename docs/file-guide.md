# Repo File Guide

这份文档按当前仓库的实际结构，说明“每个主要文件夹 / 每个主要文件”的用途。  
原则：

- 重点解释需要人维护的源码、文档、测试、模型产物。
- 对 `.git`、`__pycache__`、`.venv`、`.pytest_cache` 这类工具或缓存目录，只说明用途，不逐个展开内部文件。
- 对模型 `pkl` 和 Playwright 产物，会说明“它们是什么”和“什么时候会被覆盖”。

---

## 1. 根目录

### `app.py`
- 根启动入口。
- 作用是兼容旧命令 `streamlit run app.py`。
- 内部通过 `runpy` 转去执行 `frontend/streamlit_app/main.py`。

### `README.md`
- 项目对外说明书。
- 介绍项目目的、目录结构、运行方式、训练命令、兼容导入方式。

### `Agent.md`
- 仓库内的辅助说明文件。
- 主要给代理或协作流程使用，不参与程序运行。

### `pyproject.toml`
- Python 工程元信息。
- 定义项目名、版本、依赖、打包时包含哪些包。

### `requirements.txt`
- 根级依赖清单。
- 本地安装运行项目时优先使用这个文件。

### `.gitignore`
- Git 忽略规则。
- 用来排除缓存文件、模型产物、临时目录、虚拟环境等不应提交的内容。

---

## 2. 核心 Python 包：`ovarian_prediction/`

这个目录是后端核心逻辑，主要负责预处理、训练、预测和临床建议。

### `ovarian_prediction/__init__.py`
- 包入口。
- 对外重新导出常用类和函数，如 `OvarianPreprocessor`、`OvarianMLSystem`、`ClinicalDecisionSystem`。

### `ovarian_prediction/requirements.txt`
- 向后兼容的依赖入口。
- 当前通过 `-r ../requirements.txt` 复用根级依赖。

### `ovarian_prediction/predict.py`
- 兼容层文件。
- 保留旧导入路径 `from ovarian_prediction.predict import OvarianPredictor`。
- 实际逻辑已经拆到 `ovarian_prediction/inference/`。

### `ovarian_prediction/clinical_system.py`
- 兼容层文件。
- 保留旧导入路径 `from ovarian_prediction.clinical_system import ClinicalDecisionSystem`。
- 实际逻辑已经拆到 `ovarian_prediction/clinical/`。

### `ovarian_prediction/train.py`
- 兼容层命令入口。
- 保留旧命令 `python -m ovarian_prediction.train`。
- 实际 CLI 逻辑在 `ovarian_prediction/training/cli.py`。

---

## 3. 配置目录：`ovarian_prediction/config/`

### `ovarian_prediction/config/__init__.py`
- 配置模块对外导出入口。
- 汇总路径常量和全局常量。

### `ovarian_prediction/config/constants.py`
- 放通用常量。
- 目前主要包括默认随机种子和四个模型名。

### `ovarian_prediction/config/paths.py`
- 放路径常量。
- 定义仓库根目录、前端目录、图片目录、模型产物目录、旧模型目录等。

---

## 4. 预处理目录：`ovarian_prediction/preprocessing/`

这个目录负责“从原始表到模型输入”的全部预处理逻辑。

### `ovarian_prediction/preprocessing/__init__.py`
- 预处理子包入口。
- 对外导出常用预处理函数和类。

### `ovarian_prediction/preprocessing/feature_sets.py`
- 放特征列定义。
- 包括：
  - 需要删除的原始列
  - POR/HOR 的基础特征
  - 促排干预特征
  - 分类列
  - 目标列名

### `ovarian_prediction/preprocessing/loaders.py`
- 负责读取原始 Excel 数据。
- 把对象列转成 `category`，方便后续处理。

### `ovarian_prediction/preprocessing/cleaners.py`
- 负责清理无用列。
- 对应旧版 R 预处理里的列裁剪逻辑。

### `ovarian_prediction/preprocessing/splitters.py`
- 负责训练集 / 测试集分层切分。
- 保证 POR/HOR 标签在训练集和测试集中的比例基本一致。

### `ovarian_prediction/preprocessing/encoders.py`
- 负责分类变量 One-hot 编码。
- 把字符串分类特征转成模型能接受的数值列。

### `ovarian_prediction/preprocessing/imputers.py`
- 定义 `MICEImputer`。
- 用 `IterativeImputer + RandomForestRegressor` 做缺失值插补。

### `ovarian_prediction/preprocessing/pipeline.py`
- 定义 `OvarianPreprocessor`。
- 把删列、切分、插补、编码这些步骤串成一个完整预处理管线。

### `ovarian_prediction/preprocessing/synthetic.py`
- 生成假数据。
- 用于没有真实 Excel 时，快速训练演示模型或跑测试。
- 注意：它只负责“造一批模拟训练样本”，不直接参与线上预测。

---

## 5. 模型目录：`ovarian_prediction/models/`

这个目录负责模型训练、评估和持久化。

### `ovarian_prediction/models/__init__.py`
- 模型子包入口。
- 导出 `XGBSubmodel`、`OvarianMLSystem`、`brier_score`。

### `ovarian_prediction/models/metrics.py`
- 放模型训练/评估用的小工具。
- 主要包括：
  - `split_xy`：拆分特征和标签
  - `brier_score`：Brier 分数

### `ovarian_prediction/models/xgboost_model.py`
- 定义单个 XGBoost 子模型 `XGBSubmodel`。
- 职责包括：
  - 调参
  - 训练
  - 对齐预测特征
  - 评估
  - 保存 / 加载

### `ovarian_prediction/models/system.py`
- 定义 `OvarianMLSystem`。
- 管理四个子模型：
  - `PORDM`
  - `HORDM`
  - `PORSM`
  - `HORSM`
- 负责批量训练、批量评估、批量保存和加载。

---

## 6. 推理目录：`ovarian_prediction/inference/`

这个目录负责单患者推理。

### `ovarian_prediction/inference/__init__.py`
- 推理子包入口。

### `ovarian_prediction/inference/patient_adapter.py`
- 把单患者字典转成单行 DataFrame。
- 负责缺失值占位、分类列标准化、编码对齐。

### `ovarian_prediction/inference/predictor.py`
- 定义 `OvarianPredictor`。
- 接收患者参数，调用四个子模型，返回各模型风险概率。

---

## 7. 临床逻辑目录：`ovarian_prediction/clinical/`

这个目录负责把模型输出转成临床可读建议。

### `ovarian_prediction/clinical/__init__.py`
- 临床子包入口。
- 对外导出风险阈值、建议函数、`ClinicalDecisionSystem`、`run_demo`。

### `ovarian_prediction/clinical/thresholds.py`
- 放临床阈值和剂量范围常量。
- 比如 POR/HOR 风险分层阈值、FSH 剂量区间。

### `ovarian_prediction/clinical/rules.py`
- 放规则函数。
- 比如：
  - 风险等级判断
  - 促排方案推荐
  - FSH 剂量建议
  - LH 是否支持
  - 预估获卵数

### `ovarian_prediction/clinical/reporting.py`
- 把结果格式化成文字报告。
- 负责生成中文报告正文和 ASCII 概率条。

### `ovarian_prediction/clinical/decision_system.py`
- 定义 `ClinicalDecisionSystem`。
- 是临床层主入口。
- 把患者数据、模型结果、规则建议和报告组装在一起。

### `ovarian_prediction/clinical/demo.py`
- 放演示模式 `run_demo()`。
- 用典型患者例子打印两份完整报告。

---

## 8. 训练目录：`ovarian_prediction/training/`

### `ovarian_prediction/training/__init__.py`
- 训练子包入口。

### `ovarian_prediction/training/orchestrator.py`
- 训练编排器。
- 接收一个 DataFrame，串起预处理、训练、评估，返回预处理器、模型系统和评估结果。

### `ovarian_prediction/training/cli.py`
- 命令行训练入口。
- 负责解析参数，决定用真实数据还是假数据，保存训练产物。

---

## 9. 前端目录：`frontend/`

这个目录负责 Streamlit 页面和前端辅助逻辑。

### `frontend/__init__.py`
- 前端包入口。

---

## 10. Streamlit 应用目录：`frontend/streamlit_app/`

### `frontend/streamlit_app/__init__.py`
- Streamlit 子包入口。

### `frontend/streamlit_app/main.py`
- Streamlit 主页面脚本。
- 当前页面的大部分布局、CSS 注入、表单、结果区渲染流程都在这里调度。

---

## 11. 前端组件目录：`frontend/streamlit_app/components/`

### `frontend/streamlit_app/components/__init__.py`
- 组件包入口。

### `frontend/streamlit_app/components/layout.py`
- 页面布局辅助组件。
- 负责标题框渲染、滚动保持脚本、Figma 抓取脚本注入。

### `frontend/streamlit_app/components/metric_cards.py`
- 负责页面中的指标卡、错误条、排行卡、环形进度 SVG。

### `frontend/streamlit_app/components/result_panels.py`
- 负责两个主要结果面板：
  - 卵巢储备结果
  - 促排方案结果

---

## 12. 前端页面目录：`frontend/streamlit_app/pages/`

### `frontend/streamlit_app/pages/__init__.py`
- 页面包入口。

### `frontend/streamlit_app/pages/reserve_assessment.py`
- 卵巢储备结果页辅助渲染函数。

### `frontend/streamlit_app/pages/stimulation_planning.py`
- 促排方案结果页辅助渲染函数。

---

## 13. 前端服务目录：`frontend/streamlit_app/services/`

### `frontend/streamlit_app/services/__init__.py`
- 服务包入口。

### `frontend/streamlit_app/services/system_loader.py`
- 前端模型加载器。
- 页面启动时优先加载：
  - `artifacts/models/xgboost/`
  - 找不到再加载 `models/`
  - 仍找不到才用假数据现训演示模型

### `frontend/streamlit_app/services/upload_parser.py`
- 处理批量上传 Excel / CSV。
- 负责：
  - 模板生成
  - 列名识别
  - 患者行解析
  - 数值清洗

### `frontend/streamlit_app/services/reference_api.py`
- 对接外部参考接口。
- 同时包含本地趋势估算逻辑，如储备年龄、胚胎非整倍体风险估算。

### `frontend/streamlit_app/services/report_adapter.py`
- 前端对后端报告的轻量适配层。
- 用于提取 POR/HOR 概率等页面直接使用的数据。

---

## 14. 前端状态目录：`frontend/streamlit_app/state/`

### `frontend/streamlit_app/state/__init__.py`
- 状态包入口。

### `frontend/streamlit_app/state/session_state.py`
- Streamlit Session State 初始化逻辑。
- 用来给页面状态提供默认值。

---

## 15. 前端工具目录：`frontend/streamlit_app/utils/`

### `frontend/streamlit_app/utils/__init__.py`
- 工具包入口。

### `frontend/streamlit_app/utils/formatting.py`
- 页面格式化工具。
- 包括数字解析、显示格式化、输入框文本转换。

### `frontend/streamlit_app/utils/media.py`
- 页面媒体资源工具。
- 负责：
  - 图片转 data URI
  - 背景图模糊
  - fallback SVG 背景
  - logo 裁切与生成

---

## 16. 前端资源目录：`frontend/streamlit_app/assets/`

### `frontend/streamlit_app/assets/images/`
- 放页面图片资源。

#### `frontend/streamlit_app/assets/images/brand/`
- 品牌标识图目录。

##### `frontend/streamlit_app/assets/images/brand/fuyou_logo_raw.png`
- 原始 logo 源图。
- `media.py` 可从它裁切出圆形 logo。

##### `frontend/streamlit_app/assets/images/brand/fuyou_logo_fullseal.png`
- 页面标题区当前使用的完整 logo 图。

##### `frontend/streamlit_app/assets/images/brand/fuyou_logo_mark.png`
- 品牌图标版本。
- 当前前端代码里未直接使用，但保留为可复用素材。

#### `frontend/streamlit_app/assets/images/background/`
- 背景图目录。

##### `frontend/streamlit_app/assets/images/background/fuyou.png`
- 当前页面背景图来源。

### `frontend/streamlit_app/assets/templates/`
- 放上传模板等前端静态模板文件。

#### `frontend/streamlit_app/assets/templates/.gitkeep`
- 保持空目录被 Git 跟踪。

### `frontend/streamlit_app/assets/styles/`
- 预留给独立 CSS 样式文件。
- 当前主要样式还在 `main.py` 里。

#### `frontend/streamlit_app/assets/styles/.gitkeep`
- 保持空目录被 Git 跟踪。

---

## 17. 模型与产物目录：`artifacts/`

这个目录放运行或训练后产生的文件。

### `artifacts/models/`
- 存放模型产物。

#### `artifacts/models/xgboost/`
- 当前四个 XGBoost 子模型默认保存目录。

##### `artifacts/models/xgboost/PORDM.pkl`
- 卵巢低反应诊断模型。

##### `artifacts/models/xgboost/HORDM.pkl`
- 卵巢高反应诊断模型。

##### `artifacts/models/xgboost/PORSM.pkl`
- 卵巢低反应策略模型。

##### `artifacts/models/xgboost/HORSM.pkl`
- 卵巢高反应策略模型。

##### `artifacts/models/xgboost/.gitkeep`
- 保持空目录被 Git 跟踪；当前目录已有实际模型文件。

### `artifacts/preprocessors/`
- 存放预处理器产物。

#### `artifacts/preprocessors/mice/`
- 当前 MICE 插补器保存目录。

##### `artifacts/preprocessors/mice/porsm_imputer.pkl`
- POR 方向插补器。

##### `artifacts/preprocessors/mice/horsm_imputer.pkl`
- HOR 方向插补器。

##### `artifacts/preprocessors/mice/.gitkeep`
- 保持目录被 Git 跟踪。

### `artifacts/demo/`
- 存放验证和演示产物。

#### `artifacts/demo/.gitkeep`
- 保持目录被 Git 跟踪。

#### `artifacts/demo/playwright_smoke.png`
- 首次 Playwright 烟测截图。

#### `artifacts/demo/playwright_smoke_result.json`
- 首次烟测结构化结果。

#### `artifacts/demo/playwright_validation_strict.png`
- 严格验证截图。

#### `artifacts/demo/playwright_validation_strict.json`
- 严格验证的原始 JSON 结果。

#### `artifacts/demo/playwright_validation_strict.md`
- 严格验证的文字版步骤与结果说明。

---

## 18. 旧模型兼容目录：`models/`

这个目录是旧版模型保存位置。  
当前前端仍兼容读取它，但优先级低于 `artifacts/models/xgboost/`。

### `models/PORDM.pkl`
- 旧路径下的 PORDM 模型文件。

### `models/HORDM.pkl`
- 旧路径下的 HORDM 模型文件。

### `models/PORSM.pkl`
- 旧路径下的 PORSM 模型文件。

### `models/HORSM.pkl`
- 旧路径下的 HORSM 模型文件。

### `models/porsm_imputer.pkl`
- 旧路径下的 POR 插补器。

### `models/horsm_imputer.pkl`
- 旧路径下的 HOR 插补器。

---

## 19. 文档目录：`docs/`

### `docs/architecture.md`
- 项目总体架构说明。
- 解释后端、前端、产物目录的职责分层。

### `docs/module-map.md`
- 模块级说明。
- 比 `architecture.md` 更细，说明每个模块做什么。

### `docs/file-guide.md`
- 当前这份文件。
- 用来解释每个文件夹和主要文件的用途。

### `docs/notes/`
- 放笔记类文档。

#### `docs/notes/NOTE.md`
- 项目过程中的简短备注。

### `docs/research-archive/`
- 归档历史研究资料，不直接参与当前 Python 程序运行。

#### `docs/research-archive/.gitignore`
- 历史资料目录下的忽略规则。

#### `docs/research-archive/r-notebooks/`
- 原始 R Markdown 研究脚本目录。

##### `GLM_lasso.Rmd`
- Lasso 广义线性模型实验记录。

##### `GLM_ridge.Rmd`
- Ridge 广义线性模型实验记录。

##### `ML system (part 1).Rmd`
- 原始机器学习系统研究第 1 部分。

##### `ML system (part 2).Rmd`
- 原始机器学习系统研究第 2 部分。

##### `MLP.Rmd`
- 多层感知机实验记录。

##### `More plots.Rmd`
- 辅助图表和对比图实验。

##### `Preprocessings.Rmd`
- 原始 R 版预处理流程说明。

##### `Rondom Forest.Rmd`
- 随机森林实验记录。

##### `SVM_RBF.Rmd`
- RBF 核 SVM 实验记录。

##### `XGBoost and feature selection.Rmd`
- XGBoost 与特征筛选实验记录。

#### `docs/research-archive/source-docs/`
- 原始说明性文档。

##### `Initial covariates.docx`
- 初始候选协变量说明文档。

---

## 20. 测试目录：`tests/`

### `tests/conftest.py`
- Pytest 全局配置。
- 负责把仓库根目录加入 `sys.path`，确保测试能正确导入项目包。

### `tests/fixtures/`
- 放测试夹具数据。

#### `tests/fixtures/.gitkeep`
- 保持空目录被 Git 跟踪。

### `tests/integration/`
- 集成测试目录。

#### `tests/integration/test_pipeline_e2e.py`
- 端到端流程测试。
- 验证：合成数据 -> 预处理 -> 训练 -> 预测 -> 临床结果。

### `tests/unit/`
- 单元测试目录。

#### `tests/unit/test_preprocessing_units.py`
- 预处理单测。
- 测 `encode_categoricals` 和 `MICEImputer`。

#### `tests/unit/test_models_units.py`
- 模型单测。
- 测单个 XGBoost 子模型是否能训练和输出概率。

#### `tests/unit/test_inference_units.py`
- 推理单测。
- 测 `OvarianPredictor` 是否能返回四个概率结果。

#### `tests/unit/test_clinical_rules.py`
- 临床规则单测。
- 测风险分级和剂量建议规则。

#### `tests/unit/test_upload_parser.py`
- 上传解析单测。
- 测表头标准化和列名映射逻辑。

---

## 21. 脚本目录：`scripts/`

### `scripts/.gitkeep`
- 预留脚本目录占位文件。
- 当前还没有正式脚本落在这里。

---

## 22. 运行生成目录和缓存目录

这些目录不是核心源码，不建议手工编辑。

### `.git/`
- Git 仓库内部数据。
- 记录提交历史、对象、引用和仓库配置。

### `.venv/`
- 本地虚拟环境。
- 放 Python 解释器和安装的包。

### `.pytest_cache/`
- Pytest 运行缓存。
- 用于加速测试或记录上次测试状态。

### `.playwright-check/`
- 早期 Playwright 截图目录。
- 里面的图片是页面调试过程中的中间截图：
  - `top.png`
  - `mid.png`
  - `bottom.png`
  - `popover.png`

### `__pycache__/`
- Python 字节码缓存目录。
- 里面的 `.pyc` 文件是自动生成的，不是源码。

### 各子目录中的 `__pycache__/`
- 作用同上。
- 比如 `frontend/streamlit_app/__pycache__/`、`ovarian_prediction/models/__pycache__/` 等。

---

## 23. 一句话阅读建议

如果你只是想快速理解项目，建议按这个顺序看：

1. `README.md`
2. `docs/architecture.md`
3. `docs/module-map.md`
4. `frontend/streamlit_app/main.py`
5. `ovarian_prediction/preprocessing/pipeline.py`
6. `ovarian_prediction/models/system.py`
7. `ovarian_prediction/inference/predictor.py`
8. `ovarian_prediction/clinical/decision_system.py`
