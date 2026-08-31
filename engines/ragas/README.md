# RAG 测评胶水层（双后端）

平台 RAG 测评模块的内置引擎层：**ragas 只负责打分，本目录的胶水脚本负责与被测 RAG 拼接**。
替代云端 rag-eval（LightRAG 专用胶水），支持两种后端一键切换。

## 双后端

| | LightRAG (`lightrag`) | RAGFlow (`ragflow`) |
|---|---|---|
| 服务 | `rag/mingzhu-rag` :9621 | `rag/ragflow` 栈 :8180 |
| 检索模式 | naive / local / global / hybrid / mix / bypass(裸LLM对照) | rf-vector(纯向量) / rf-hybrid(向量+词项) / rf-rerank(加权+重排) |
| 答案来源 | LightRAG 自身 /query 生成 | 胶水用 OPENAI_* 生成（带引用标注） |
| 上下文来源 | /query only_need_context 解析 | /api/v1/retrieval chunks |
| 单题延迟 | ~70-140s (GLM 建图/关键词) | 秒级检索 + ~60s 生成 |

切换入口：RAG 页「⚙ 公共检索参数 → 被测 RAG 后端」（存 module_config.json `rag.backend`，
默认 ragflow）。两后端的模式数据可**共存**于模式对比雷达，同题集直接横向对比。

## 文件

```
scripts/retrieve.py      题集 × 模式 → data/eval_<mode>.jsonl (+ .meta.json 参数指纹)
scripts/evaluate.py      ragas 17 指标打分 → results/scores_cache.json 断点缓存
                         + scores_<mode>.json + eval_report.md
scripts/synth_questions.py  AI 合成题集（平台「AI 合成」按钮）
data/eval_questions.jsonl   题集（问题 + reference 要点，可网页增删）
```

## 契约（与平台 app.py 对齐，零侵入聚合）

- 模式 = `data/eval_*.jsonl` 文件名自动发现（`eval_questions` 除外）
- 分数缓存 key = `{mode}@{答卷md5前8}:{qid}:{metric}`，重新检索后旧分自动失效
- 参数指纹 = 检索参数 + 题集 md5（retrieve.py 与 app.py 同算法），改参数/题集 → 全部模式过期

## 手跑

```bash
RAG_BACKEND=ragflow  ../../.venv-ragas/bin/python scripts/retrieve.py rf-vector
../../.venv-ragas/bin/python scripts/evaluate.py rf-vector --limit 2   # 冒烟限题
../../.venv-ragas/bin/python scripts/evaluate.py --with-noise          # 补噪声敏感度
```

裁判模型：`module_config.json` rag.judge_model > rag.model > .env `OPENAI_MODEL`（默认 glm-4.7）。
embedding 走宿主机 MLX（bge-m3，零 API 费用，容器内经 `EVAL_EMB_URL` 指向 host.docker.internal:7997）。

## 已知边界

- RAGFlow `keyword=true`（LLM 查询改写的混合检索）需要给 ragflow 配默认 chat model，本栈未配 →
  不设该模式，rf-hybrid 用 `vector_similarity_weight=0.5` 的加权混合替代。
- ragflow 登录：RSA 公钥内置于镜像（各安装同一把），胶水自动换 token，全程无人工参与。
- LightRAG 一题两跳（先取上下文再生成答案），延迟约两倍；bypass 模式属于 lightrag 后端。
