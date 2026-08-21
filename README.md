# 测评聚合平台

> 定位:**只做聚合层**。四大开源引擎负责测评本身,平台负责数据集管理、任务编排、结果汇总、对比可视化。不重造指标轮子。

## 引擎分工

| 引擎 | 负责维度 |
|---|---|
| **ragas** | RAG 端到端质量(忠实度/相关性/检索指标 33 项)+ 智能体指标(ToolCallAccuracy 等) |
| **OpenCompass** | 模型能力基准,六基准本地化 + 限题控费 |
| **Langfuse** | 智能体轨迹采集 + 云端监控 |
| **promptfoo** | 安全红队 + 攻击用例在线配置 |

## LLM 基准口径(2026-08 换代)

- **回归口径**:C-Eval(52 学科)/ CMMLU(67 学科)/ MMLU(57 学科),每子集前 N 题截断控费
- **前沿口径**:GPQA(diamond 全量 198 题)/ **AIME 2025**(30 题全量,整数精确匹配判卷,与公开榜口径一致)/ **MMLU-Pro**(每类前 20 题 ×14 类)
- GSM8K/HellaSwag 已退役:前沿模型 95-99% 饱和,无区分度
- 实测校验:deepseek-v4-flash AIME 2025 = 96.7%、gpt-4o-mini = 10%,与 Artificial Analysis / LLM Stats 公开榜位置吻合

## 功能

- 📊 总览:四引擎接入状态 + 评测历史与趋势图(题级实时进度)
- 📚 RAG 测评:13 种检索模式对比 / 逐题明细 / 数据集与题集管理 / 评测报告
- 🎓 LLM 基准:模型登记(任意 OpenAI 兼容端点 + 思考强度五档)/ 成绩矩阵 / 学科雷达 / 运行历史
- 🤖 智能体测评:deepagents 被测智能体 + 轨迹生成 + ragas agent 指标 + Langfuse 云轨迹
- 🛡️ 安全红队:promptfoo 攻击扫描 / 用例在线 CRUD / 漏洞清单
- 📖 指标词典:33 项指标的计算方法/输入依赖/方向/踩坑记录,前端悬浮可查

## 快速开始

```bash
pip install fastapi uvicorn httpx psutil   # 平台本体
python app.py                               # http://127.0.0.1:8800
```

LLM 基准模块需另装 OpenCompass venv、RAG 模块依赖 `D:\job\rag-eval` 数据源,
完整步骤见 [DEPLOY.md](DEPLOY.md),总体设计见 [PLAN.md](PLAN.md)。

## 约定

- 引擎产物一律落"目录 + 标准文件名",平台零侵入聚合(`runs/` 下,不入库)
- 后台任务:子进程 + 日志文件 + 平台侧进程检测,防重复启动
- 前端单文件无构建(static/index.html),ECharts CDN
- 密钥不入库:`.env` / `module_config.json` / `runs/llm/models.json` 均在 .gitignore 中,模板见 `*.example`
