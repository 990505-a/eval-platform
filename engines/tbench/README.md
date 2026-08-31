# Terminal-Bench 2 引擎(线A · 任务完成度基准)

业界终端智能体基准(89 题): 每题独立 Docker 容器内真实干活, 任务完成后**容器内跑测试判分**
(客观, 不靠裁判)。官方 harness = Harbor;智能体 = 内置 **Terminus-2**(榜单同款官方参考实现,
原生支持 `api_base` 指向任意 OpenAI 兼容端点), 分数可与公开榜横向对比。

## 文件

- `run_bench.py` — 平台跑批入口:
  - `--oracle --limit 1`:零 API 费用的环境预检(官方参考解)
  - `--model <登记名> [--limit N] [--include glob,...]`:用 LLM 页登记的模型跑 Terminus-2
    (include 逗号分隔多个 glob;注意任务全名带 `terminal-bench/` 前缀, 要用 `*schemelike*` 而非 `schemelike*`)
  - 解析逐任务 result.json → `runs/tbench/latest.json` + `history.jsonl` + `archive/`
- `seed_taskenv.sh` — 判分器 uv 预置(一次性, 见下)
- 原始产物(轨迹/逐任务明细)在 `runs/tbench/jobs/<job>/`

## 判分器 uv 预置(防"冤案 0 分")

TB 判分脚本普遍从 github 下载 uv/uvx(`astral.sh/uv/0.9.5`), `uvx -p 3.13` 还要再从
github 下载托管 CPython —— 国内经 Clash 到 release-assets.githubusercontent.com 间歇
SSL 断链, 下载失败 → `uvx: command not found` → **测试根本没跑, 直接 0 分**(2026-08-25
schemelike 智能体 48 步解出仍被判 0 的实锤根因)。

```
./seed_taskenv.sh     # 预置 uv/uvx + uv 认的 cpython-3.13 到 runs/tbench/taskroot/
```

run_bench.py 自动把 taskroot 窄挂载进任务容器的 `/root/.local/bin` 与
`/root/.local/share/uv`(两处子路径, 不遮蔽镜像其他内容): 官方安装器照常跑(失败无害),
之后 `source env` / `uvx` 用的都是预置件, pytest 走 PyPI 直连白名单。关闭: `TBENCH_SEED_UV=0`。
TB 判分升级 uv 钉版时重跑 seed 脚本(`TBENCH_UV_VERSION=<版本> ./seed_taskenv.sh`, 先清
`taskroot/share/uv/python/`)。

## 环境

```
python -m venv .venv-tbench
.venv-tbench/bin/pip install harbor     # Windows 为 .venv-tbench\Scripts\pip
```

需要 Docker daemon(任务容器由它起; Docker 部署平台时 compose 已挂 docker.sock, 任务容器
作为兄弟容器运行)。国内网络首次拉任务镜像较慢, runner 已放宽环境超时 3 倍;
**不要**给 Docker Desktop 配手动代理(实测会卡死后端, 保持 system 模式)。

## 手跑示例

```
.venv-tbench/bin/python engines/tbench/run_bench.py --oracle --limit 2
.venv-tbench/bin/python engines/tbench/run_bench.py --model Ornith-1.5-35B-A3B-MLX-4bit --limit 4
```
