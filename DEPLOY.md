# 测评聚合平台 · 部署说明

> 定位：聚合层平台（FastAPI 单文件后端 + 单文件前端），四大开源引擎负责测评本身。
> 本包已内置六大基准的本地化数据集（`engines/opencompass/data`，约 35M），LLM 基准模块开箱可跑。

## 模块可用性（本包范围）

| 模块 | 引擎 | 本包状态 |
|---|---|---|
| 🎓 LLM 基准 (C-Eval/CMMLU/MMLU/GPQA/AIME 2025/MMLU-Pro) | OpenCompass | ✅ 按下面步骤装好 venv 即可用 |
| 🛡️ 安全红队 | promptfoo | ⚠️ 装好 npm 依赖后可用，但靶机是智能体服务（见下） |
| 📚 RAG 测评 / 🤖 智能体测评 | ragas + deepagents + Langfuse | ❌ 依赖 `rag-eval` 目录（未包含在本包，需单独获取） |

平台启动不依赖 rag-eval——缺它时 RAG/智能体页签为空，其余正常。

## 环境要求

- Windows（脚本里有 win 特有处理，如 taskkill 杀进程树）
- Python 3.12（opencompass 0.5.3 实测环境）
- Node.js ≥ 18（仅红队模块需要 promptfoo）

## 部署步骤

### 1. 配置密钥（必做）

```
copy .env.example .env
copy module_config.json.example module_config.json
```

- `.env`：平台默认的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`（OpenAI 兼容端点均可，如智谱/DeepSeek）
- `module_config.json`：rag / agent / redteam 三个模块可各自配独立的模型与密钥，
  未配置的字段运行时回退 `.env`。`base_url` 填到 `/v1` 或 `/v4` 这一级，**不要以反斜杠结尾/包含 `\`**。
- LLM 基准的被测模型不走 module_config：启动平台后在「🎓 LLM 基准 → 模型管理」里登记
  （模型名 + Base URL + API Key + 思考强度），存 `runs/llm/models.json`。

改配置入口也可以在网页里：⚙️ 设置 / 模型设置（`module_config.json` 即其后端存储）。

### 2. 平台基础环境（跑 app.py）

```
python -m venv .venv-platform
.venv-platform\Scripts\pip install fastapi uvicorn httpx psutil
.venv-platform\Scripts\python app.py
```

打开 http://127.0.0.1:8800 （端口用环境变量 `PORT` 覆盖）。

### 3. LLM 基准模块（OpenCompass）

```
python -m venv .venv-opencompass
.venv-opencompass\Scripts\pip install opencompass==0.5.3
```

- 数据集已随包附带，无需下载；每次运行会自动生成"限题副本 + lite 配置"控费（每子集前 N 题）。
- 在 🎓 LLM 基准页先点「预检」验证模型名/密钥/端点，再跑基准。
- 并发默认 2 worker × 1 qps（低 QPS 账号安全值），上调可能导致 429。

### 4. 红队模块（promptfoo）

```
npm install
```

- 攻击用例在 `engines/promptfoo/cases.json`（网页红队页签也可在线增删）。
- 扫描靶机是智能体服务的 `/chat`（默认 http://127.0.0.1:8820）——该服务在 `rag-eval` 里，
  没有它红队扫描无法发起（平台会提示"靶机未启动"）。

### 5. RAG / 智能体模块（可选，需 rag-eval）

平台通过 `RAGAS_DIR` 环境变量找 rag-eval（默认 `D:\job\rag-eval`）。如果拿到了 rag-eval 目录，
放到任意路径后设置 `RAGAS_DIR` 指向它即可，还需要其内部三个独立 venv（见 rag-eval 自己的 README）。

## 已知注意事项（源码实测踩坑提炼）

- `base_url` 含 `\` 会导致请求路径变 `%5C` 被 404，平台已做校验会直接报错提示。
- C-Eval 判卷用自定义智能答案提取（`engines/opencompass/smart_eval.py`），
  官方"取第一个大写字母"的口径会把英文作答的模型误判到接近 0 分，勿改回官方版。
- 历史成绩在 `runs/`（本包附带了示例结果），删掉不影响运行。
- 手动跑基准：`.venv-opencompass\Scripts\python engines\opencompass\run_bench.py <模型名> <基准key>`，
  基准 key：ceval / cmmlu / mmlu / gpqa / aime2025 / mmlu_pro。

## 目录速览

```
app.py                     平台后端（全部 API）
static/index.html          前端单文件（hash 路由, 8 页面）
DEPLOY.md                  本文件
PLAN.md                    设计文档（模块分工/里程碑/约定）
metrics_dict.json          指标词典（前端📖页数据源）
module_config.json.example 各模块模型配置模板
.env.example               平台默认密钥模板
engines/opencompass/       OpenCompass 编排脚本 + 本地化数据集(35M) + lite 配置
engines/promptfoo/         红队攻击用例库
runs/                      运行产物（成绩矩阵/历史/归档, 附示例）
```
