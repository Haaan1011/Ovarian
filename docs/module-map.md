# Module Map

## Backend

- `ovarian_prediction/config/`
  路径常量和全局常量。
- `ovarian_prediction/preprocessing/`
  原始数据读取、清洗、编码、插补、切分与合成数据生成。
- `ovarian_prediction/models/`
  XGBoost 子模型、训练评估工具和四模型系统封装。
- `ovarian_prediction/inference/`
  单患者输入适配与概率预测。
- `ovarian_prediction/clinical/`
  风险阈值、规则函数、报告格式化与临床决策入口。
- `ovarian_prediction/training/`
  训练编排与命令行入口。

## Frontend

- `frontend/streamlit_app/main.py`
  页面总调度。
- `frontend/streamlit_app/components/`
  指标卡、结果面板、布局辅助。
- `frontend/streamlit_app/services/`
  模型加载、上传解析、参考接口、报告适配。
- `frontend/streamlit_app/state/`
  Session State 默认值初始化。
- `frontend/streamlit_app/utils/`
  格式化与媒体资源辅助函数。
- `frontend/streamlit_app/assets/`
  页面图片、模板和样式占位目录。

## Docs And Tests

- `docs/file-guide.md`
  文件级说明索引。
- `docs/notes/NOTE.md`
  简短过程记录。
- `docs/research-archive/`
  历史研究材料归档。
- `tests/`
  单元测试与集成测试。
