# 测评聚合平台 · 部署说明

> 定位：聚合层平台（FastAPI 单文件后端 + 单文件前端），开源引擎负责测评本身。
> 本包已内置六大基准的本地化数据集（`engines/opencompass/data`，约 35M），LLM 基准模块开箱可跑。

## 模块可用性（本包范围）

| 模块 | 引擎 | 本包状态 |
|---|---|---|
| 🎓 LLM 基准 (C-Eval/CMMLU/MMLU/GPQA/AIME/MMLU-Pro) | OpenCompass | ✅ 按下面步骤装好 venv 即可用 |
| 🛡️ 安全红队 | promptfoo | ⚠️ 装好 npm 依赖后可用，但靶机是智能体服务（见下） |
| 📚 RAG 测评 | ragas（内置胶水层 `engines/ragas`） | ✅ **双后端可切换**：RAGFlow（本地栈） / LightRAG（rag-mingzhu），见下 |
| 🤖 智能体测评 | deepagents + DeepEval + Langfuse | ✅ **本包自包含**: 本地任务集跑轨迹 → DeepEval 双口径评分; Langfuse 配了 key 可加轨迹查看与对话抽样评分 |

RAG 测评已内置胶水层（ragas 只打分，`engines/ragas/scripts/retrieve.py` 负责与被测 RAG 拼接）：
RAG 页「⚙ 公共检索参数」里切换后端（默认 ragflow，模式 rf-vector/rf-hybrid/rf-rerank；
lightrag 为 naive/local/global/hybrid/mix/bypass），两后端同题集结果可共存对比。
云端 rag-eval 目录仍可用 `RAGAS_DIR` 环境变量指回（存在即优先）。

## 环境要求

- Windows（脚本里有 win 特有处理，如 taskkill 杀进程树）
- Python 3.12（opencompass 0.5.3 实测环境）
- Node.js ≥ 18（仅红队模块需要 promptfoo）

## 部署步骤

### 0. Docker 一键部署（推荐，Mac/Linux/Windows 通用）

```
cp .env.example .env              # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
cp module_config.json.example module_config.json
docker compose up -d --build      # 首次构建约 10 分钟 (opencompass 含 torch, 约 5-6GB 镜像)
```

打开 http://localhost:8800 。镜像内已含全部引擎（OpenCompass venv + Node 20 + promptfoo），
不用再装 venv/npm。`docker compose build` 走国内镜像源直连，并在 compose 里清空了
Docker Desktop 注入的代理（若宿主机代理只监听 127.0.0.1，注入进容器只会导致全部请求超时）。

- 配置实时生效：宿主机编辑 `.env` / `module_config.json` / `engines/promptfoo/cases.json` 即可，已挂载进容器
- 成绩持久化：`runs/` 目录挂载在宿主机，`docker compose build` 重建不丢
- 平台代码改动后重建：`docker compose up -d --build`（依赖层有缓存，只重打代码层，很快）

> 源码已做跨平台适配：venv 路径（Scripts\python.exe ↔ bin/python）、promptfoo.cmd ↔ promptfoo、
> `HOST` 环境变量覆盖监听地址（容器内为 0.0.0.0），Windows 本地 venv 启动方式不受影响。

### 0.1 被测基础设施（本仓库自带）

```
rag/mingzhu-rag/        名著RAG (lightrag-hku, 四大名著+莎士比亚全集)   → localhost:9621 (WebUI /webui)
rag/ragflow/            对比RAG (RAGFlow v0.27 + ES, 同样的 MLX 模型)   → localhost:8180
rag/mlx-models/         MLX 本地推理: embedding(bge-m3)+rerank(Qwen3)   → localhost:7997 (宿主机原生, Metal)
langfuse/langfuse.env   本地 Langfuse 接入配置 (平台与被测智能体共用)
```

**被测智能体不再内置**（2026-09 重构）：自定义智能体测评与红队扫描的靶机是
**你自己的智能体服务**——按平台「🤖 智能体测评 → ② 讲解 · 接入指南」实现 HTTP 契约
（POST /run 必须, /files /file /chat 可选, 指南里有 50 行参考实现）, 起在宿主机任意端口,
然后在平台页签里填地址(默认 `http://127.0.0.1:8820`, Docker 部署用 `AGENT_SVC_URL` 覆盖)。
**MLX 服务跑在宿主机**（不在 Docker 里）：`cd rag/mlx-models && ./start.sh`，Mac 重启后需手动启一次；
embedding/rerank 因此零 API 费用，LLM（建图/对话）仍走云端 key。

**LLM 用智谱 GLM coding plan 时**（`.env` 已配置）：端点是
`https://open.bigmodel.cn/api/coding/paas/v4`（**不是**标准 `api/paas/v4`——coding 套餐额度只在
coding 端点生效，标准端点会报 1113 余额不足），模型名 `glm-4.7`（思考型；`glm-4.7-flash` 常年
1305 过载）。国内直连无需代理。智谱 coding 端点 QPS 低，rag-mingzhu 已配并发 2 + 超时 600s。
若换回 Kimi coding plan（api.kimi.com 国内不通），需在宿主机跑 `scripts/proxy-bridge.py`
（0.0.0.0:7898 → Clash 7897）并给容器配 http_proxy，参考 git 历史里的 kimi-proxy 配置块。

**RAG 入库**（产生真实 API 费用，先小额验证）：

```
docker compose exec rag-mingzhu python ingest.py --dry-run     # 统计分片
docker compose exec rag-mingzhu python ingest.py --pieces 2    # 每本 2 片冒烟
docker compose exec rag-mingzhu python ingest.py               # 全量
```

**本地 Langfuse（替代云端）**：

```
docker compose --profile langfuse up -d   # 首次拉镜像较久; postgres/clickhouse/minio/redis + web/worker
```

- 管理界面 http://localhost:3000 ，账号为 compose 里 `LANGFUSE_INIT_USER_EMAIL` 配置的邮箱（密码为 `.env` 的
  `LF_ADMIN_PASSWORD`；Langfuse 全部密钥经 `.env` 的 `LF_*` 变量注入，模板见 `.env.example`），
  组织/项目/API key 已通过 `LANGFUSE_INIT_*` 自动创建
- 平台与智能体读 `langfuse/langfuse.env`（host 用容器网络地址 `http://langfuse-web:3000`），
  智能体配置了 key 即自动上报轨迹；平台的 ⚙️设置/Langfuse 也读这份文件
- 不用时 `docker compose --profile langfuse down`（数据在命名卷里，不丢）

### 1. 配置密钥（必做，非 Docker 部署时）

```
copy .env.example .env
copy module_config.json.example module_config.json
```

- `.env`：平台默认的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`（OpenAI 兼容端点均可，如智谱/DeepSeek）
- `module_config.json`：四个模块（llm / agent / redteam / rag）可各自配独立的模型与密钥，
  未配置的字段运行时回退 `.env`。`base_url` 填到 `/v1` 或 `/v4` 这一级，**不要以反斜杠结尾/包含 `\`**。

改配置入口也可以在网页里：⚙️ 设置 / 模型设置（`module_config.json` 即其后端存储）。

### 2. 平台基础环境（跑 app.py）

```
python -m venv .venv-platform
.venv-platform\Scripts\pip install fastapi uvicorn httpx psutil python-multipart
.venv-platform\Scripts\python app.py
```

打开 http://127.0.0.1:8800 （端口用环境变量 `PORT` 覆盖）。

### 3. LLM 基准模块（OpenCompass）

```
python -m venv .venv-opencompass
.venv-opencompass\Scripts\pip install opencompass==0.5.3
```

- 数据集已随包附带，无需下载；每次运行会自动生成"限题副本 + lite 配置"控费（每子集前 N 题）。
- **基准口径（2026-08 换代）**：回归口径 C-Eval/CMMLU/MMLU（每子集前 N 题）+ 前沿口径
  GPQA（diamond 全量 198 题）/ AIME 2025（30 题全量，`\boxed{}` 整数精确匹配判卷，公开榜同口径）/
  MMLU-Pro（每类前 20 题 ×14 类）；BBH/GSM8K/HellaSwag 已退役（前沿模型 95-99% 饱和，无区分度）。
- 模型登记支持独立 Base URL / API Key / 思考强度五档（default/off/low/medium/high，
  off=关思考最快最省；网关不支持时自动回退默认）。
- 在 🎓 LLM 基准页先点「预检」验证模型名/密钥/端点，再跑基准。
- 并发默认 2 worker × 1 qps（低 QPS 账号安全值），上调可能导致 429。

### 4. 红队模块（promptfoo）

```
npm install
```

- 攻击用例在 `engines/promptfoo/cases.json`（网页红队页签也可在线增删）。
- 扫描靶机是智能体服务的 `/chat`（默认 http://127.0.0.1:8820）——该服务在 `rag-eval` 里，
  没有它红队扫描无法发起（平台会提示"靶机未启动"）。

### 4.5 智能体基准矩阵(Harbor)

```
python -m venv .venv-tbench
.venv-tbench/bin/pip install harbor        # Windows 为 .venv-tbench\Scripts\pip
```

- 用法在网页 🤖 智能体测评 → 「🏁 智能体基准」页签:下拉选基准(TB 2.0/2.1/Pro · τ³-bench ·
  GAIA · SWE-bench Verified) → 换新基准先「🧪 oracle 预检」(零 API 费用验证环境) →
  选 LLM 页登记的模型 → 限题数「▶ 跑基准」; 历史轮次按 基准×模型 对比。
- 智能体 = Terminus-2(榜单同款官方实现, Harbor 内置), 自动带 api_base 指向模型登记的
  OpenAI 兼容端点; 判分 = 任务容器内跑测试, 分数可与公开榜横向对比。
- Docker 部署: compose 已挂载 docker.sock, 任务容器由宿主机 daemon 起(兄弟容器), 需重建镜像
  (`docker compose up -d --build`)。
- **国内网络注意**: 任务镜像从 Docker Hub 拉取较慢(走已配置的镜像源), 首次跑某任务可能要
  几分钟——runner 已把环境超时放宽 3 倍; 切勿改 Docker Desktop 手动代理配置(实测会卡死后端
  启动, 保持 system 模式即可)。

#### 🎬 实时演示(跑基准时看智能体干活)

- **终端直播**: 跑批时「🏁 智能体基准」页出现实时面板, 左侧 xterm.js 终端实时渲染 Terminus-2
  在任务容器里敲的每条命令(平台 tail `agent/recording.cast` asciinema 增量流), 右侧 ATIF
  轨迹步骤 + token 实时计数(读 `agent/trajectory.json`); 默认自动跟随运行中的任务, 下拉可手动切换。
- **完整回放**: 任务 chips 上的 ▶ 按钮弹层播放整段终端录像(asciinema-player, 可调倍速);
  跑完的任意一轮都能回放。
- **harbor view**: 「🔍 harbor view」按钮拉起 Harbor 官方 web 轨迹浏览器(浏览全部 jobs),
  Docker 部署需 compose 里的 `8080-8089` 端口映射(已配)。
- **历史轮次回看与清理**: 演示面板左侧轮次下拉选任意历史轮次(含重试产生的孤儿轮次)回看;
  🕘 历史轮次表每行 🗑 删除该轮记录(连带任务目录/归档), 轮次下拉旁 🗑 只删轮次目录;
  跑批进行中禁止删除。
- 机制: 录像是 Terminus-2 智能体的能力(与数据集无关, TB 系/GAIA/SWE-bench 通用);
  **oracle 不录像**; 换其他 agent 前先确认它是否写 recording.cast(源码里只有 terminus-2
  和 qwen_code 录)。τ³-bench 是工具对话型, 首次跑建议先 oracle 预检确认产物结构。

### 5. RAG / 智能体模块

智能体测评（DeepEval 双口径, Docker 部署时镜像内已含 .venv-agenteval, 跳过安装）：

```
python -m venv .venv-agenteval
.venv-agenteval/bin/pip install -r engines/deepeval/requirements.txt   # Windows 为 .venv-agenteval\Scripts\pip
```

- 被测智能体 = 你自己的服务(接入契约见平台「智能体测评 → 接入指南」); 任务集在平台侧 `engines/deepeval/tasks/default.jsonl`
- 「生成轨迹」→ run_tasks.py 逐题执行并采集文件/记忆终态证据 → `runs/agenteval/trajectories-*.jsonl`
- 「计算指标」→ evaluate.py: DeepEval 裁判 2 项（走 ⚙️设置/模块设置-agent 的模型）+ 确定性核验 4 项
- 零成本自测: `run_tasks.py --selftest` + `evaluate.py --selftest`（不调 API）
- Langfuse 对话抽样评分: 「Langfuse 轨迹」页签 → 🔬 抽样评分（拉最近轨迹 → DeepEval 打分 → 分数写回）

RAG 测评（ragas 胶水层）：

```
python -m venv .venv-ragas
.venv-ragas\bin\pip install -r engines/ragas/requirements.txt   # Windows 为 .venv-ragas\Scripts\pip
```

- 被测后端二选一（网页可切换）：RAGFlow 栈 `docker compose -f rag/ragflow/docker-compose.yml -p eval-platform-ragflow up -d`
  （:8180，数据集 `mingzhu-test`）；或 LightRAG（compose 里的 `rag-mingzhu` :9621）
- embedding/重排走宿主机 MLX（:7997，`cd rag/mlx-models && ./start.sh`，Mac 重启后需手动启）
- 若拿到云端 rag-eval 目录，设 `RAGAS_DIR` 指向它即回退云端胶水（智能体轨迹评测 eval_agent.py 也随它）

### 4.7 自有评测集(题库 / 任务集 / 自定义基准)

三层入口, 全部照 RAG 的"AI 合成 → 待审 → 人工采纳"流程, 已内置示例可直接体验:

- **LLM 自有题库**（LLM 页 →「自有题库」区域）：选择题库 `engines/opencompass/data/own/own.jsonl`，可在网页直接添加/删除题目；业务资料可直接上传 `.txt/.md`，勾选一个或多个素材后点击「生成选中」或「生成全部」；平台写入 `engines/opencompass/data/own_seeds/`，再生成待审核题目。Docker Compose 已将素材目录和 `own.jsonl` 挂载到宿主机，重建容器不会丢失。
- **智能体任务集**(🤖 页 →「② 考题 · 任务集」页签): 任务集 `engines/deepeval/tasks/default.jsonl` 在线增删 + 单题试跑
  (服务未启动也可管理); AI 合成依据 `engines/deepeval/task_seeds.md`——「工具清单 + 领域素材」,
  **换你自己的项目改这份文件即可**; 合成任务带 expect_tools/expect_file/expect_memory_contains
  等标注与审核理由, 采纳后进正式任务集, 「📏 自家打分」测的就是它。建议新任务先在「体验」页单题试跑确认可解。
- **智能体自定义基准**(🤖 页 → 基准下拉「🧩 自定义数据集」): 填本地任务目录(Harbor 自动切 `-p`)或
  registry 名。自建任务四件套 `task.yaml / solution.sh / run-tests.sh / tests/`,
  先「oracle 预检」免费验证任务可解再上模型。

## 已知注意事项（源码实测踩坑提炼）

- `base_url` 含 `\` 会导致请求路径变 `%5C` 被 404，平台已做校验会直接报错提示。
- C-Eval 判卷用自定义智能答案提取（`engines/opencompass/smart_eval.py`），
  官方"取第一个大写字母"的口径会把英文作答的模型误判到接近 0 分，勿改回官方版；
  思考型模型输出含 `</think>` 时取最终段、显式答案标记取最后一次出现（草稿里的中间假设不算）。
- run_bench 启动时会探测网关：temperature（Kimi 系只接受 1.0）与思考强度参数（被拒自动回退），
  并统一带浏览器 UA（部分网关的 Cloudflare 封 python-urllib 默认 UA，403 error 1010）。
- 历史成绩在 `runs/`（本包附带了示例结果），删掉不影响运行。
- 手动跑基准：`.venv-opencompass\Scripts\python engines\opencompass\run_bench.py <模型名> <基准key>`
  （Linux/Mac 为 `.venv-opencompass/bin/python`），
  基准 key：ceval / cmmlu / mmlu / gpqa / aime2025 / mmlu_pro。

## 目录速览

```
app.py                     平台后端（全部 API）
static/index.html          前端单文件（操作页）
static/skill/eval-platform-guide/  可下载的 AI 使用 Skill
DEPLOY.md                  本文件
PLAN.md                    设计文档（模块分工/约定）
metrics_dict.json          引擎指标原始词典（前端不作为主入口）
module_config.json.example 各模块模型配置模板
.env.example               平台默认密钥模板
engines/opencompass/       OpenCompass 编排脚本 + 本地化数据集
engines/deepeval/          外部智能体任务跑批与评分
engines/tbench/            Harbor 通用智能体基准
engines/promptfoo/         红队攻击用例库
runs/                      运行产物（本地生成）
```

## AI 使用 Skill

平台智能体页“接入”面板提供 `eval-platform-guide` 下载入口（页面内不再设指引页，操作指引统一通过该 Skill 提供）：

- `/static/skill/eval-platform-guide/SKILL.md`
- `/static/skill/eval-platform-guide/README.md`

该 Skill 不包含密钥或运行数据，安装后可让 AI 指导平台部署、配置、运行评测和排障。
