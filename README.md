# 测评聚合平台

> 定位:**只做聚合层**。开源引擎负责测评本身,平台负责数据集管理、任务编排、结果汇总、对比可视化。不重造指标轮子。

## 引擎分工

| 引擎 | 负责维度 |
|---|---|
| **ragas** | RAG 端到端质量(忠实度/相关性/检索指标 33 项) |
| **OpenCompass** | 模型能力基准,六基准本地化 + 限题控费 |
| **DeepEval** | 自定义智能体测评(第②部分):接入你自己的智能体 → ToolCorrectness/TaskCompletion 裁判 + 确定性核验 |
| **Harbor** | 通用智能体测评(第①部分):TB 2.0/2.1/Pro · τ³-bench · GAIA · SWE-bench Verified,容器内测试判分,分数对标公开榜 |
| **Langfuse** | 智能体轨迹采集 + 云端/本地监控 |
| **promptfoo** | 安全红队 + 攻击用例在线配置 |

## LLM 基准口径(2026-08 换代)

- **回归口径**:C-Eval(52 学科)/ CMMLU(67 学科)/ MMLU(57 学科),每子集前 N 题截断控费
- **前沿口径**:GPQA(diamond 全量 198 题)/ **AIME 2025**(30 题全量,整数精确匹配判卷,与公开榜口径一致)/ **MMLU-Pro**(每类前 20 题 ×14 类)
- GSM8K/HellaSwag 已退役:前沿模型 95-99% 饱和,无区分度
- 实测校验:deepseek-v4-flash AIME 2025 = 96.7%、gpt-4o-mini = 10%,与 Artificial Analysis / LLM Stats 公开榜位置吻合

## 功能

- 📊 总览：模块状态、运行进度和最近记录
- 📚 RAG：选择后端、重新检索、运行评测、管理题集和查看答卷
- 🎓 LLM：登记模型、预检、运行基准和查看结果
- 🤖 智能体：接入外部服务、维护任务、生成轨迹和运行评测
- 🛡️ 红队：维护攻击用例、发起扫描和查看漏洞结果
- 🧩 自有评测集：LLM 自有题库、智能体任务集和 Harbor 自定义任务

## 快速开始

```bash
pip install fastapi uvicorn httpx psutil python-multipart   # 平台本体
python app.py                                             # http://127.0.0.1:8800
```

LLM 基准模块需另装 OpenCompass venv，RAG 模块可使用内置双后端胶水层；完整步骤见 [DEPLOY.md](DEPLOY.md)。

操作指引统一通过 `eval-platform-guide` Skill 提供（智能体页“接入”面板有下载入口），安装到支持的 AI 客户端后，可让 AI 按本项目事实指导部署、配置、运行和排障。直接资源地址：`/static/skill/eval-platform-guide/SKILL.md`。

LLM 页面的“自有题库”支持直接上传 `.txt/.md` 业务资料，上传后可勾选一个或多个素材，点击“生成选中”或“生成全部”出题；素材会保存到 `engines/opencompass/data/own_seeds/`，无需手动复制文件。

## 约定

- 引擎产物一律落“目录 + 标准文件名”，平台负责编排和聚合（`runs/` 为运行产物）
- 后台任务：子进程 + 日志文件 + 平台侧进程检测，防重复启动
- 前端为单文件静态页面，无构建步骤
- 密钥不入库：`.env` / `module_config.json` / `runs/llm/models.json` 均在 `.gitignore` 中，模板见 `*.example`
