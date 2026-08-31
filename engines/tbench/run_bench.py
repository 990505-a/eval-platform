# -*- coding: utf-8 -*-
"""智能体基准跑批 (Harbor 官方 harness): harbor run 包装 → 解析成绩 → runs/tbench/

线A 引擎(任务完成度基准): 智能体 = Terminus-2(榜单同款官方实现, 原生支持任意 OpenAI
兼容端点), 基准从 Harbor 注册表(276+ 数据集)任选, 换 --dataset 即换基准。
被测模型复用平台 LLM 页的模型登记(runs/llm/models.json)。

可用基准(Harber 注册表名, 详见 hub.harborframework.com/datasets):
    terminal-bench/terminal-bench-2-1      TB 2.1 修订版(默认, 修了 28 题判分)
    terminal-bench/terminal-bench-2        TB 2.0
    terminal-bench-pro/terminal-bench-pro  TB Pro(200 题, 更难更贵)
    sierra-research/tau3-bench             τ³ 工具对话(375 题, pass^k 口径)
    gaia/gaia                              GAIA 通用助理(165 题)
    swe-bench/swe-bench-verified           SWE-bench Verified(500 题, 最硬最贵)

用法(独立手跑):
    python run_bench.py --oracle --limit 1                       # 零成本环境预检
    python run_bench.py --model <登记名> [--dataset <名>] [--limit N] [--include glob]
    --dataset 也可传本地任务目录(自动切 harbor run -p): 自建基准写 task.yaml/
    solution.sh/run-tests.sh/tests/ 四件套, 先 --oracle 验证可解再上模型

产物契约(与平台对齐):
    runs/tbench/latest.json      最近一轮成绩 {bench, model, n, passed, pass_rate, tasks[]}
    runs/tbench/history.jsonl    每轮一行
    runs/tbench/archive/         归档
    runs/tbench/jobs/            Harbor 原始产物(轨迹/逐任务明细)
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = ROOT / "runs" / "tbench"
# jobs 目录: 容器内运行时用宿主机绝对路径(compose 注入 TBENCH_JOBS_DIR + 同路径挂载),
# 否则 harbor 生成的任务容器 bind-mount 源路径宿主 daemon 找不到; 本地直跑时就是 RUNS/jobs
JOBS = Path(os.environ.get("TBENCH_JOBS_DIR") or RUNS / "jobs")
MODELS = ROOT / "runs" / "llm" / "models.json"
HARBOR = ROOT / ".venv-tbench" / ("Scripts/harbor.exe" if os.name == "nt" else "bin/harbor")
# 默认 TB 2.1(2026-08 修订版: 修复 28 题判分问题, 榜单已切此口径)
DATASET = os.environ.get("TBENCH_DATASET", "terminal-bench/terminal-bench-2-1")
# 平台思考档位 → Terminus-2 reasoning_effort
EFFORT = {"off": "none", "low": "low", "medium": "medium", "high": "high"}


def _load_env_file():
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())



def _pick_task_proxy() -> str:
    """任务容器内下载(pip/apt)走的代理: 显式 TBENCH_TASK_PROXY 优先(="" 禁用);
    自动探测 —— Clash 允许局域网时 7897 优先(无桥进程依赖), 7898 桥兜底。
    探测在本机回环与容器视角都试, 但返回值统一用容器可达的 host.docker.internal 地址。"""
    env = os.environ.get("TBENCH_TASK_PROXY")
    if env is not None:
        return env
    import socket

    def ok(h: str, p: int) -> bool:
        try:
            socket.create_connection((h, p), timeout=2).close()
            return True
        except OSError:
            return False

    for port in (7897, 7898):
        if ok("host.docker.internal", port) or ok("127.0.0.1", port):
            return f"http://host.docker.internal:{port}"
    return ""


def find_model(name: str) -> dict:
    if not MODELS.exists():
        raise SystemExit(f"模型登记不存在: {MODELS}(先在 LLM 基准页登记模型)")
    for m in json.loads(MODELS.read_text(encoding="utf-8")):
        if m.get("model") == name:
            return m
    raise SystemExit(f"模型未登记: {name}(在 LLM 基准页 → 模型管理 里添加, 复用同一份登记)")


def parse_jobs(jobs_dir: Path, since: datetime) -> dict | None:
    """找本轮新生成的 job, 逐任务目录解析 result.json
    (结构: task_name / verifier_result.rewards.reward / exception_info)。"""
    for job in sorted(jobs_dir.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
        if datetime.fromtimestamp(job.stat().st_mtime) < since:
            continue
        tasks = []
        for tr in sorted(job.glob("*/result.json")):
            try:
                d = json.loads(tr.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            rew = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            tasks.append({
                "task": str(d.get("task_name") or d.get("trial_name") or tr.parent.name).split("/")[-1],
                "passed": rew is not None and float(rew) > 0,
                "reward": rew,
                "exception": bool(d.get("exception_info")),
            })
        if tasks:
            return {"job": job.name, "results": tasks}
    return None


def save_outcome(bench: str, model: str, agent: str, parsed: dict | None, log_tail: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if parsed:
        tasks = parsed["results"]
        passed = sum(1 for t in tasks if t["passed"])
        rec = {"ts": ts, "bench": bench, "model": model, "agent": agent, "job": parsed["job"],
               "n": len(tasks), "passed": passed,
               "pass_rate": round(passed / len(tasks), 4), "tasks": tasks}
        note = ""
    else:
        rec = {"ts": ts, "bench": bench, "model": model, "agent": agent, "n": 0, "passed": 0,
               "pass_rate": None, "tasks": [],
               "error": "未解析到成绩, 看 runs/tbench/log.txt 与 jobs 目录"}
        note = log_tail[-500:]
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "latest.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    with (RUNS / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "tasks"},
                           ensure_ascii=False) + "\n")
    arch = RUNS / "archive" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    arch.parent.mkdir(exist_ok=True)
    arch.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[tbench] {bench} 通过率 {rec['pass_rate']} ({rec['passed']}/{rec['n']}){note}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="平台模型登记名(LLM 基准页)")
    ap.add_argument("--dataset", default=DATASET, help="Harbor 注册表数据集名(默认 TB 2.1)")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("TBENCH_LIMIT") or 4))
    ap.add_argument("--include", default=os.environ.get("TBENCH_INCLUDE"),
                    help="任务名过滤 glob(逗号分隔多个, 如 schemelike*,torch-tensor-parallelism)")
    ap.add_argument("--timeout-mult", default=None, type=float,
                    help="任务超时倍数(平台前端传入; 默认 TBENCH_TIMEOUT_MULT 或 2)")
    ap.add_argument("--concurrency", default=None, type=int,
                    help="并发任务数(平台前端传入; 默认 TBENCH_N 或 1)")
    ap.add_argument("--oracle", action="store_true", help="跑参考解, 零 API 费用(环境预检)")
    args = ap.parse_args()
    if not args.oracle and not args.model:
        ap.error("需要 --model <登记名> 或 --oracle")

    _load_env_file()
    if not HARBOR.exists():
        raise SystemExit(f"harbor 不存在: {HARBOR}(python -m venv .venv-tbench && "
                         ".venv-tbench/bin/pip install harbor)")

    _tp = _pick_task_proxy()
    _pip_idx = os.environ.get("TBENCH_PIP_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple")
    # 判分器 uv 预置: TB 判分脚本普遍先从 github 下载 uv/uvx(+uvx -p 3.13 再拉托管 CPython),
    # 国内经 Clash 到 release-assets.githubusercontent.com 间歇 SSL 断链 → "uvx: command not found"
    # → 测试根本没跑 → 冤案 0 分(2026-08-25 schemelike 实锤)。taskroot 预置 uv/uvx/env 与
    # uv 托管 CPython 3.13, 窄挂载进 /root/.local 的两个子路径: 官方安装器照常跑(失败无害),
    # source env / uvx 用到的是预置件, pytest 等包从 PyPI 直连白名单拉。生成: seed_taskenv.sh
    # 双路径: 存在性检查走平台进程可见路径(RUNS); 挂载行写宿主机路径(任务容器由宿主 daemon
    # 起, bind-mount 源必须是宿主绝对路径) —— 与 jobs/pipcache 的同路径挂载策略一致
    _taskroot_seen = RUNS / "taskroot"
    _taskroot_host = JOBS.parent / "taskroot"
    _seed_uv = ((_taskroot_seen / "bin" / "uv").exists()
                and os.environ.get("TBENCH_SEED_UV", "1") != "0")
    # 直连白名单(实测容器直连这些 CDN 又快又稳, 反倒是 Clash 链路 1/3 概率 SSLError/超长尾):
    # pip 三件套(pypi.org/files.pythonhosted.org/download.pytorch.org) + 镜像源 + debian apt,
    # 其余外网(github/huggingface 等)才走 Clash 代理
    _DIRECT = ("localhost,127.0.0.1,host.docker.internal",
               "pypi.org", "files.pythonhosted.org", "download.pytorch.org",
               "deb.debian.org", "archive.ubuntu.com",
               "security.ubuntu.com")
    # 例外(2026-08-25 实测): debian/ubuntu 官方源直连有"整晚坏"的时段(apt update 3 分钟仍失败,
    # 连 kv-store 都死在装 tmux), 而走 Clash 稳定 72s 装完 —— 给 apt 单独挂按域名代理配置
    # (apt.conf.d 优先级高于环境代理, 只影响 apt, 不动 pip 的直连优化; 镜像 base 无关)
    _APT_PROXY_HOSTS = ("deb.debian.org", "security.debian.org",
                        "archive.ubuntu.com", "security.ubuntu.com")
    if _tp or _pip_idx:
        _mirror_host = _pip_idx.split("//")[-1].split("/")[0] if _pip_idx else ""
        _no_proxy = ",".join(x for x in (*_DIRECT, _mirror_host) if x)
        # 共享 pip 轮子缓存: 同一包第二轮起秒装(判分器的 2GB CUDA 全家桶只需痛一次)。
        # 卷源必须是宿主机路径(任务容器由宿主 daemon 起), 与 jobs 同目录同路径挂载策略。
        _pipcache = Path(os.environ.get("TBENCH_JOBS_DIR") or RUNS / "jobs").parent / "pipcache"
        _pipcache.mkdir(parents=True, exist_ok=True)
        RUNS.mkdir(parents=True, exist_ok=True)
        env_lines = []
        if _pip_idx:
            env_lines += [f"      - PIP_INDEX_URL={_pip_idx}", "      - PIP_DISABLE_PIP_VERSION_CHECK=1"]
        if _tp:
            env_lines += [f"      - HTTP_PROXY={_tp}", f"      - HTTPS_PROXY={_tp}"]
        env_lines += [f"      - NO_PROXY={_no_proxy}"]
        vol_lines = [f"    volumes:", f"      - {_pipcache}:/root/.cache/pip"]
        if _seed_uv:  # 窄挂载两个子路径, 不遮蔽镜像里 /root/.local 的其他内容
            vol_lines += [f"      - {_taskroot_host}/bin:/root/.local/bin",
                          f"      - {_taskroot_host}/share/uv:/root/.local/share/uv"]
        if _tp:  # apt 官方源走代理(直连有整晚坏的时段, 见 _APT_PROXY_HOSTS 注释)
            (_taskroot_seen / "apt-proxy.conf").write_text("".join(
                f'Acquire::{sch}::Proxy::{h} "{_tp}";\n'
                for h in _APT_PROXY_HOSTS for sch in ("http", "https")), encoding="utf-8")
            vol_lines.append(
                f"      - {_taskroot_host}/apt-proxy.conf:/etc/apt/apt.conf.d/99tbench-apt-proxy:ro")
        (RUNS / "overlay-proxy.yaml").write_text(
            "services:\n  main:\n    environment:\n" + "\n".join(env_lines) + "\n" + "\n".join(vol_lines) + "\n",
            encoding="utf-8")
    # dataset 是本地任务目录时用 -p(Harbor 直读目录, 自建基准入口), 注册表名用 -d;
    # parse_jobs/save_outcome 只认 Harbor 产物契约, 对两种来源无差别
    _ds_flag = "-p" if Path(args.dataset).is_dir() else "-d"
    cmd = [str(HARBOR), "run", _ds_flag, args.dataset, "-o", str(JOBS), "--yes",
           "-l", str(args.limit),
           # 并发: 本地单 GPU 模型(oMLX)默认串行——两个长上下文智能体会互相饥饿
           # (实测 write-compressor 被连发中的 torch 饿了 7.5 分钟烧穿超时);
           # 云端 API 无此限制, 前端可选 2-4 提速
           "-n", str(args.concurrency or int(os.environ.get("TBENCH_N", "1"))),
           # 任务容器代理: 用 compose overlay 把代理写进容器环境(覆盖装 tmux/torch 的环境准备阶段,
           # --ae/--ve 再兜一层给智能体/判分会话); 关闭: TBENCH_TASK_PROXY=""
           *([ "--extra-docker-compose", str(RUNS / "overlay-proxy.yaml")]
             if (tp := _pick_task_proxy()) else []),
           # 注意 --ae/--ve 的 NO_PROXY 必须与 overlay 一致(完整直连白名单), 否则会覆盖回小名单
           *(sum(([f, f"HTTP_PROXY={tp}", f, f"HTTPS_PROXY={tp}",
                   f, f"NO_PROXY={_no_proxy if _tp else 'localhost,127.0.0.1,host.docker.internal'}"]
                  for f in ("--ae", "--ve")), [])
             if tp else []),
           # 国内网络拉任务镜像偏慢: 环境构建/启动超时放宽 3 倍, 避免白跑
           "--environment-build-timeout-multiplier", "3",
           # agent 会话准备(装 tmux/asciinema 的 apt 命令, 基线 120s)在重镜像下不够用:
           # 2026-08-25 torch(CUDA 镜像, x86 翻译)实测 120s 装完即死, 模型根本没上场 → 同样 ×3
           "--agent-setup-timeout-multiplier", "3",
           # 全局任务超时倍数(命令/判分等): 1×=公开榜同口径(线上模型对标), 2×=本地慢模型补偿
           "--timeout-multiplier",
           str(args.timeout_mult if args.timeout_mult else os.environ.get("TBENCH_TIMEOUT_MULT", "2"))]
    if args.include:  # 逗号分隔多任务(glob), 各成一个 -i(harbor 支持重复传, 便于重跑失败题)
        for pat in args.include.split(","):
            if pat.strip():
                cmd += ["-i", pat.strip()]
    agent, model_str = "oracle", "oracle"
    if args.oracle:
        cmd += ["-a", "oracle"]
    else:
        m = find_model(args.model)
        os.environ["OPENAI_API_KEY"] = m.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        model_str = m["model"]
        agent = "terminus-2"
        cmd += ["-a", "terminus-2", "-m", f"openai/{model_str}",
                "--ak", f"api_base={m['base_url'].rstrip('/')}"]
        eff = EFFORT.get(m.get("think") or "")
        if eff:
            cmd += ["--ak", f"reasoning_effort={eff}"]

    started = datetime.now()
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"[tbench] {agent} × {model_str} · {args.dataset} · 限 {args.limit} 题 · "
          f"{'预检(免费)' if args.oracle else '真实推理(计费)'}", flush=True)
    print("[tbench] " + " ".join(cmd), flush=True)
    env = {k: v for k, v in os.environ.items() if k not in ("HOST", "PORT")}
    # 网络现实(实测): harbor 数据集域名(supabase.co, Cloudflare CDN)在国内直连是"间歇通",
    # 走 Clash 桥反而稳定不通 → 默认直连 + 外层重试兜底; 需要时 TBENCH_PROXY=http://x 强制代理。
    proxy = os.environ.get("TBENCH_PROXY", "")
    if proxy:
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            env[k] = proxy
    noproxy = ["host.docker.internal", "127.0.0.1", "localhost"]
    for base in (os.environ.get("OPENAI_BASE_URL", ""),):
        host = base.split("//")[-1].split("/")[0].split(":")[0]
        if host:
            noproxy.append(host)
    env["NO_PROXY"] = env["no_proxy"] = ",".join(dict.fromkeys(noproxy))
    tp = _pick_task_proxy()
    print(f"[tbench] 代理: harbor自身={proxy or '(直连+重试)'} · 任务容器内={tp or '⚠ 无可用代理(容器内下载可能超时)'}", flush=True)
    print(f"[tbench] 判分器预置: {'✓ uv+cpython-3.13 已挂载(' + str(_taskroot_host) + ')' if _seed_uv else '✗ 无 taskroot 预置(github 断链时 uvx 类判分可能冤案 0 分, 见 seed_taskenv.sh)'}", flush=True)
    rc = 1
    for attempt in range(1, 5):
        with (RUNS / "log.txt").open("a", encoding="utf-8") as logf:
            logf.write(f"\n[platform] {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}"
                       f"  (attempt {attempt}/4)\n")
            logf.flush()
            rc = subprocess.call(cmd, cwd=str(HERE), stdout=logf, stderr=subprocess.STDOUT, env=env)
        if rc == 0:
            break
        print(f"[tbench] harbor 退出码 {rc}, 重试 {attempt}/4(数据集域名间歇被掐, 多试能过)…", flush=True)
        import time as _t
        _t.sleep(6)
    print(f"[tbench] harbor 最终退出码 {rc}, 解析成绩…", flush=True)
    parsed = parse_jobs(JOBS, started)
    save_outcome(args.dataset, model_str, agent, parsed,
                 (RUNS / "log.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
