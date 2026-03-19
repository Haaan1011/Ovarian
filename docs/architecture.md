# Architecture

本项目分为三层：

1. `ovarian_prediction/`
后端核心，负责预处理、模型训练、推理和临床建议。

2. `frontend/streamlit_app/`
Streamlit 前端，负责页面布局、表单输入、结果展示和前端辅助服务。

3. `artifacts/` 与 `models/`
运行和训练产物目录，分别承载当前默认模型路径与旧版兼容模型路径。

当前入口关系：

- `app.py` 是根启动入口，兼容 `streamlit run app.py`
- `frontend/streamlit_app/main.py` 是前端主页面
- `ovarian_prediction/train.py` 是训练命令兼容入口

整体数据流：

1. 原始患者数据进入 `preprocessing/`
2. 训练阶段由 `training/` 编排并调用 `models/`
3. 推理阶段由 `inference/` 调用四个子模型
4. 临床输出由 `clinical/` 转成规则建议与中文报告
