# 测评聚合平台 · 总体计划

> 定位：**只做聚合层**。开源引擎负责测评本身，平台负责数据集管理、任务编排、结果汇总、对比可视化。
> 原则：不重造指标轮子；引擎产物落成"目录 + 标准文件名"约定，平台零侵入聚合。

## 一、引擎分工（定稿）

| 引擎 | 负责维度 | 接入方式 | 里程碑 |
|---|---|---|---|
| **ragas** | RAG 端到端质量 | 双后端胶水层内置(`engines/ragas`): RAGFlow(本地·rf-vector/hybrid/rerank) ⇄ LightRAG(naive/local/global/hybrid/mix/bypass) 网页可切换, 同题集共存对比; 云端 rag-eval 仍可经 `RAGAS_DIR` 指回 | M1 ✅ |
| **OpenCompass** | 模型能力基准(知识/推理/数学) | 六大基准本地化+限题控费: 回归口径 C-Eval/CMMLU/MMLU(每子集前N题) + 前沿口径 GPQA/AIME 2025/MMLU-Pro(2026-08 换代, GSM8K/HellaSwag 因饱和退役)；模型登记 + 运行历史 | M2 ✅ |
| **Harbor (TB2)** | 通用智能体测评(第①部分): 容器内测试判分, 与公开榜同口径 | 内置引擎 `engines/tbench`: run_bench.py 包装 `harbor run`, 被测模型复用 LLM 页登记, oracle 预检零成本, 产物落 runs/tbench/ | M3 ✅ |
| **DeepEval** | 自定义智能体测评(第②部分): 接入你自己的智能体 → ToolCorrectness/TaskCompletion 裁判 + 确定性核验 4 项 | 内置引擎 `engines/deepeval`: run_tasks.py 按接入契约(POST /run)跑平台侧任务集(tasks/default.jsonl)生成轨迹+终态证据 → evaluate.py 双口径评分(2026-09 重构: 删除内置名著智能体, 平台零内置靶机) | M3 ✅ |
| **Langfuse** | 智能体轨迹采集 + 线上监控 | 云端/本地 API 代理 + 轨迹本地落盘(runs/agenteval/) + DeepEval 抽样评分闭环(score_chats.py: 拉轨迹→打分→写回) | M3 ✅ |
| **promptfoo** | 安全红队 + 用例在线配置 | 子进程跑 `promptfoo eval`(cases.json 动态生成 yaml)，解析 JSON 输出 | M4 ✅ |

## 二、前端信息架构（本次交付骨架）

```
📊 总览          五引擎接入状态卡 + RAG 运行进度(live) + 评测历史与趋势图
📚 RAG 测评      [M1·已接通] 模式对比/逐题明细/数据集/题集管理/实时日志/评测报告
🎓 LLM 基准      [M2·已接通] 模型登记管理 / 基准任务选择 / 成绩矩阵 / 运行历史
🤖 智能体测评    [M3·已接通] 对话/文件/固定任务集 + 轨迹生成 + DeepEval 双口径指标 + Langfuse 轨迹与抽样评分
🛡️ 安全红队      [M4·已接通] 攻击用例在线配置 / 扫描结果 / 漏洞清单
⚖️ A/B 对比      不自建 —— 直接用司南(OpenCompass 排行榜/竞技场), 本页为门户指引
⚙️ 设置          引擎接入配置 / 裁判模型 / 数据源路径
```

## 三、后端 API 规划（按模块）

**已实现（M1）**：`/api/overview` `/api/scores` `/api/compare` `/api/datasets` `/api/report` `/api/log` `/api/run` `/api/metrics-map` + `/api/modules`

**规划中**：

| 模块 | 接口 | 说明 |
|---|---|---|
| M2 LLM | `GET/POST /api/llm/models` | 模型登记(本地路径/API端点) |
| | `POST /api/llm/run` | 拉起 opencompass 任务(后台进程+进度解析) |
| | `GET /api/llm/results` | 成绩矩阵(模型×基准) |
| M3 Agent | `GET /api/agent/traces` | 代理 Langfuse API，拉轨迹列表 |
| | `GET /api/agent/traces/{id}` | 单条轨迹详情(步骤/工具调用/耗时) |
| | `POST /api/agent/eval` | 对轨迹数据集跑 DeepEval 双口径指标 |
| M4 红队 | `POST /api/redteam/scan` | 拉起 promptfoo redteam |
| | `GET /api/redteam/results` | 漏洞清单/通过率 |
| M4 A/B | `POST /api/ab/run` | pairwise 对比任务 |
| | `GET /api/ab/results` | 胜率/逐对结果 |
| 设置 | `GET/POST /api/settings` | 各引擎接入配置(落 SQLite) |

## 四、里程碑

- **M1（已完成）**：RAG 聚合上线——进度监控/对比雷达/明细/数据集/日志/报告/一键启动
- **M2（已完成）**：OpenCompass 基准 + 模型登记管理(/api/llm/models) + 运行历史(/api/llm/history) + 基准参数化
- **M3（已完成）**：deepagents 被测智能体 + Langfuse 云轨迹代理 + 轨迹生成(/api/agent/generate → run_tasks.py)
  + DeepEval 双口径闭环(/api/agent/eval → evaluate.py → agent_scores.json, 2026-08 从 ragas agent 指标换代)
  + Langfuse 对话抽样评分闭环(/api/agent/chatscore → score_chats.py → 分数写回轨迹)
- **M4（已完成）**：promptfoo 红队 + 攻击用例在线配置(cases.json CRUD)；A/B 对比不自建, 用司南门户
- **M5（部分完成）**：✅ 评测历史与趋势(/api/history + 总览趋势图, RAG/LLM/红队/智能体四源汇总)；
  ✅ 指标词典(metrics_dict.json 单一事实源 + /api/metrics-dict + 前端"📖 指标词典"页 +
  各页指标名❓悬浮提示；33 项指标的计算方法/输入依赖/方向/本项目的坑, 对照 ragas 0.4.3 源码核实；
  另含 13 个 RAG 检索模式说明(retrieve.py 实验矩阵), 模式对比页 chips 悬浮可查)；
  ✅ 评测按模式选择性运行(RAG 页模式 chips 多选 + /api/run 传 modes + evaluate.py argv 过滤)；
  ⏳ 设置落 SQLite 与 CI 定时回归为可选项, 当前 .env/json 方案够用暂不做
- **M6（已完成）**：自有评测集构造(开源基准测通用能力, 自有集测业务贴合度, 复用 RAG 的
  "AI 合成→待审→人工采纳"流程)——
  ① LLM 自有题库(bench key `own`): kind=custom 手写配置 lite_own.py(CustomDataset + MCQ 模板 +
  smart_ceval_postprocess 判分, 与 C-Eval 同口径), 题库 data/own/own.jsonl 在线 CRUD,
  合成源 data/own_seeds/ 种子语料(engines/opencompass/synth_bench.py, 题带 basis 材料锚),
  est 动态题数、空题库双端拦截(run_bench + /api/llm/run);
  ② 智能体任务集: tasks.jsonl 在线 CRUD(/api/agent/tasks, 服务未启动也可管理) +
  engines/deepeval/task_seeds.md(工具清单+领域素材, 换项目改这份) + synth_tasks.py 合成
  (expect_tools 合法性过滤, expect_* 四字段与 server→run_tasks→evaluate 全链契约逐字对齐);
  ③ tbench 自定义基准: dataset 支持本地任务目录(run_bench 按是否目录切 harbor -p/-d,
  前端下拉「🧩 自定义数据集」+ dataset 输入), registry 名照旧走 -d

## 五、约定

- 引擎产物目录：`RAGAS_DIR`(已有) / `OPENCOMPASS_DIR` / `LANGFUSE_URL` / `PROMPTFOO_DIR`，均可用环境变量覆盖
- 后台任务一律：子进程 + 日志文件 + 平台侧进程检测（防重复启动），复用 M1 的模式
- 前端单文件无构建，ECharts CDN；占位模块统一"待接入"角标 + 里程碑标记
