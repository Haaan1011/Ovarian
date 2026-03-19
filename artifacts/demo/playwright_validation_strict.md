
# Playwright Strict Validation

## Target
- URL: http://127.0.0.1:8501
- PID: 1100369
- CWD: /home/zhishi/PredictOvarianResponse-main

## Process
```text
UID          PID    PPID  C STIME TTY          TIME CMD
zhishi   1100369 1083708  0 16:57 pts/16   00:00:00 /home/zhishi/anaconda3/envs/Han_Overian/bin/python /home/zhishi/anaconda3/envs/Han_Overian/bin/streamlit run app.py --server.headless true --server.port 8501
```

## Port
```text
LISTEN   0        128                   0.0.0.0:8501             0.0.0.0:*       users:(("streamlit",pid=1100369,fd=6))                                         
LISTEN   0        128                      [::]:8501                [::]:*       users:(("streamlit",pid=1100369,fd=7))
```

## Steps
- 1. 使用 Playwright 访问 http://127.0.0.1:8501 并等待首屏渲染。
- 2. 读取首页主标题并校验文本是否为“卵巢储备功能与促排卵方案AI辅助系统”。
- 3. 点击左侧“生成卵巢储备功能结果”，等待进度环和储备结果卡片出现。
- 4. 点击右侧“生成促排卵方案结果”，等待风险条和方案指标卡片出现。
- 5. 记录页面截图，并检查是否存在 console error / page error / failed request。


## Checks
- latest_code_started_successfully: True
- correct_port_8501: True
- page_opened_successfully: True
- title_correct: True
- console_has_no_errors: True
- reserve_flow_complete: True
- plan_flow_complete: True
- no_horizontal_layout_overflow: True

## Console Summary
- warnings: 10
- errors: 0
- page_errors: 0
- failed_requests: 0

## Screenshot
- /home/zhishi/PredictOvarianResponse-main/artifacts/demo/playwright_validation_strict.png

## Final Verdict
- all_required_checks_passed: True
