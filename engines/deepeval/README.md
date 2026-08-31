# DeepEval 智能体评测引擎(线B)

智能体轨迹质量评分引擎(接替 ragas 的 agent 指标位, ragas 收缩回纯 RAG):

| 指标 | 口径 | 来源 |
|---|---|---|
| tool_correctness | 裁判(GEval): 实际工具调用 vs 期望集合, 漏调/多调都扣分 | DeepEval |
| task_completion | 裁判(GEval): 对照指令与终态证据判任务达成度 | DeepEval |
| tool_recall | 确定性: 期望工具被调到的比例(多调不罚) | 本引擎自算 |
| answer_hit | 确定性: 回答含期望关键词比例 | 本引擎自算 |
| file_hit / memory_hit | 确定性: 期望文件/记忆真实产出(环境终态核验) | 本引擎自算 |

## 文件

- `run_tasks.py` — 任务跑批: 智能体服务逐题执行 → `runs/agenteval/trajectories-*.jsonl`
  (每题附工作区文件列表与长期记忆快照, 作终态核验证据)
- `evaluate.py` — 双口径评分 → `runs/agenteval/agent_scores.json` + `archive/` 自动归档
- `score_chats.py` — Langfuse↔DeepEval 对接: 拉最近对话轨迹 → GEval 打分 →
  分数写回 Langfuse(`deepeval-answer-quality`), 落 `runs/agenteval/chatscores.json`

## 环境

```
python -m venv .venv-agenteval
.venv-agenteval/bin/pip install -r engines/deepeval/requirements.txt   # Windows 为 Scripts\pip
```

裁判模型: ⚙️设置/模块设置-agent 的模型与密钥(未配置时回退平台 `.env` 的 OPENAI_*)。
被测智能体: `agents/mingzhu-agent`(:8820, 任务集 `tasks.jsonl` 由其自带)。

## 自测(零 API 费用)

```
.venv-agenteval/bin/python engines/deepeval/run_tasks.py --selftest   # 合成轨迹
.venv-agenteval/bin/python engines/deepeval/evaluate.py --selftest    # 确定性核验链路
```
