r"""
RAG 评测聚合平台(后端) —— 只聚合开源工具的产物, 不实现任何指标
    数据来源: D:\job\rag-eval 下的 ragas 评测脚本产物(jsonl 数据集 / scores_cache.json / evaluate.log)
运行:
    D:\job\.venv\Scripts\python.exe app.py        (默认 http://127.0.0.1:8800)
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

RAGAS_DIR = Path(os.environ.get("RAGAS_DIR", r"D:\job\rag-eval"))
# ragas 项目整理后的子目录 (data/ 评估数据, results/ 分数, logs/ 日志, scripts/ 脚本)
RAGAS_DATA = RAGAS_DIR / "data"
RAGAS_RESULTS = RAGAS_DIR / "results"
RAGAS_LOGS = RAGAS_DIR / "logs"
PORT = int(os.environ.get("PORT", "8800"))


def discover_modes() -> tuple:
    """模式从 data/eval_*.jsonl 动态发现 —— 胶水脚本产出新模式文件即自动纳入.
    eval_questions.jsonl 是题库不是模式, 排除."""
    if RAGAS_DATA.exists():
        found = sorted({p.stem.removeprefix("eval_") for p in RAGAS_DATA.glob("eval_*.jsonl")}
                       - {"questions"})
        if found:
            return tuple(found)
    return ("mix", "naive")

# 智能体服务 (deepagents, 独立 venv, 端口被系统保留时用 AGENT_SVC_URL 覆盖)
AGENT_SVC = os.environ.get("AGENT_SVC_URL", "http://127.0.0.1:8820")
AGENT_DIR = RAGAS_DIR / "agent"

BASE_DIR = Path(__file__).parent

METRIC_LABELS = {
    "faithfulness": "忠实度",
    "answer_relevancy": "回答相关性",
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "context_entity_recall": "实体召回率",
    "factual_correctness": "事实正确性",
    "answer_correctness": "回答正确性",
    "semantic_similarity": "语义相似度",
    "response_groundedness": "回答有据度",
    "noise_sensitivity": "噪声敏感度(越低越好)",
    "rougeL": "ROUGE-L",
    "bleu": "BLEU",
    "chrf": "chrF",
    "string_similarity": "字面相似度",
    "exact_match": "精确匹配",
    "rubric_accuracy": "裁判-准确性(1-5)",
    "rubric_completeness": "裁判-完整性(1-5)",
    "rubric_grounding": "裁判-有据性(1-5)",
}
# 1-5 分制的裁判指标(雷达图归一化用)
RUBRIC_SCALE = 5
# 雷达图展示的核心指标(0-1 分制, 越高越好)
RADAR_METRICS = [
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    "context_entity_recall", "factual_correctness", "answer_correctness",
    "semantic_similarity", "response_groundedness",
]

app = FastAPI(title="RAG 评测聚合平台")


# ---------- 工具函数 ----------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_cache() -> dict:
    p = RAGAS_RESULTS / "scores_cache.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # 正在被评测进程写入的瞬间可能读半截, 忽略本轮


def _find_eval_procs() -> list:
    """找到正在跑 evaluate.py 的进程(只认 python 直跑该脚本的, 避免误匹配诊断命令/编辑器)."""
    me = os.getpid()
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == me:
                continue
            cmd = proc.info["cmdline"] or []
            if (len(cmd) >= 2
                    and str(cmd[0]).lower().rsplit("\\", 1)[-1].rsplit("/", 1)[-1] in ("python.exe", "python", "pythonw.exe")
                    and str(cmd[1]).replace("\\", "/").endswith("evaluate.py")):
                found.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def eval_running() -> bool:
    """检测 evaluate.py 是否在运行。"""
    return bool(_find_eval_procs())


@app.post("/api/stop")
def stop_eval():
    """停止正在运行的评测(强杀进程; 断点缓存原子写入不受损, 可续跑)."""
    procs = _find_eval_procs()
    if not procs:
        raise HTTPException(404, "没有正在运行的评测")
    pids = []
    for proc in procs:
        try:
            proc.kill()
            pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    psutil.wait_procs(procs, timeout=5)
    return {"stopped": True, "pids": pids}


def _mode_content_fp(mode: str) -> str | None:
    """答卷文件内容指纹: 缓存 key 前缀({mode}@{fp})与其一一对应; 重新检索后旧 key 自动失效."""
    import hashlib
    p = RAGAS_DATA / f"eval_{mode}.jsonl"
    if not p.exists():
        return None
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def parse_cache(cache: dict) -> dict:
    """{mode: {qid: {metric: score}}} — 只认与当前答卷内容指纹匹配的分数(key 形如 mode@内容fp:qid:metric)。
    重新检索后答卷变 -> 内容指纹变 -> 旧分数自动不展示, 保证界面永远是当前数据口径。"""
    out: dict = {m: {} for m in discover_modes()}
    want = {}
    for m in out:
        fp = _mode_content_fp(m)
        if fp:
            want[f"{m}@{fp}"] = m
    for key, val in cache.items():
        parts = key.split(":", 2)
        if len(parts) != 3:
            continue
        head, qid, metric = parts
        mode = want.get(head)
        if not mode:
            continue
        out[mode].setdefault(qid, {})[metric] = val
    return out


# ---------- API ----------
@app.get("/api/overview")
def overview():
    datasets = {m: read_jsonl(RAGAS_DATA / f"eval_{m}.jsonl") for m in discover_modes()}
    total_q = sum(len(v) for v in datasets.values())
    cache = load_cache()
    parsed = parse_cache(cache)
    metric_names = sorted({k.rsplit(":", 1)[1] for k in cache})
    per_q = max(len(metric_names), 1)

    done_q = sum(len(v) for v in parsed.values())
    finished = (RAGAS_RESULTS / "scores_mix.json").exists() and (RAGAS_RESULTS / "scores_naive.json").exists()

    log_mtime = None
    log = RAGAS_LOGS / "evaluate.log"
    if log.exists():
        log_mtime = datetime.fromtimestamp(log.stat().st_mtime).strftime("%H:%M:%S")

    return {
        "running": eval_running(),
        "finished": finished,
        "total_questions": total_q,
        "done_questions": done_q,
        "done_items": len(cache),
        "total_items": total_q * per_q,
        "metrics": metric_names,
        "datasets": {m: len(v) for m, v in datasets.items()},
        "log_mtime": log_mtime,
    }


@app.get("/api/scores")
def scores():
    """全部缓存分数, 按模式分组, 附题目信息。"""
    parsed = parse_cache(load_cache())
    result = {}
    for mode in discover_modes():
        rows = read_jsonl(RAGAS_DATA / f"eval_{mode}.jsonl")
        info = {str(r.get("id")): r for r in rows}
        items = []
        for qid, metrics in parsed.get(mode, {}).items():
            r = info.get(qid, {})
            items.append({
                "id": qid,
                "book": r.get("book", "?"),
                "user_input": r.get("user_input", "?"),
                "metrics": metrics,
            })
        items.sort(key=lambda x: int(x["id"]) if str(x["id"]).isdigit() else 0)
        result[mode] = items
    return {"labels": METRIC_LABELS, "modes": result,
            "needs_reference": sorted(NEEDS_REFERENCE)}


@app.get("/api/compare")
def compare():
    """各检索模式各指标均值(仅统计已完成的题)。"""
    parsed = parse_cache(load_cache())
    out = {}
    for mode in discover_modes():
        agg: dict[str, list] = {}
        for qid, metrics in parsed.get(mode, {}).items():
            for name, val in metrics.items():
                if isinstance(val, (int, float)):
                    agg.setdefault(name, []).append(val)
        out[mode] = {
            name: {"mean": round(sum(v) / len(v), 4), "n": len(v)}
            for name, v in agg.items()
        }
    return {"radar_metrics": RADAR_METRICS, "labels": METRIC_LABELS, "modes": out}


@app.get("/api/datasets")
def datasets():
    return {m: read_jsonl(RAGAS_DATA / f"eval_{m}.jsonl") for m in discover_modes()}


@app.get("/api/report")
def report():
    p = RAGAS_RESULTS / "eval_report.md"
    if not p.exists():
        return {"exists": False}
    return {"exists": True, "content": p.read_text(encoding="utf-8")}


@app.get("/api/log")
def log_tail(n: int = 40):
    p = RAGAS_LOGS / "evaluate.log"
    if not p.exists():
        return {"lines": []}
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-n:]}


@app.post("/api/run")
async def run_eval(request: Request):
    """后台拉起 ragas 评测脚本(占用检测: 已在跑则拒绝)。
    body 可选 {"modes": ["mix", ...]} — 只跑指定检索模式; 不传则全跑。"""
    if eval_running():
        raise HTTPException(409, "评测已在运行中")
    script = RAGAS_DIR / "scripts" / "evaluate.py"
    if not script.exists():
        raise HTTPException(404, f"未找到 {script}")

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001  (无 body / 非 json = 全跑)
        body = {}
    modes = (body or {}).get("modes") or []
    if not isinstance(modes, list) or not all(isinstance(m, str) for m in modes):
        raise HTTPException(400, "modes 须为字符串数组")
    if modes:
        known = set(discover_modes())
        bad = [m for m in modes if m not in known]
        if bad:
            raise HTTPException(400, f"未知模式: {bad}; 可用: {sorted(known)}")

    py = RAGAS_VENV_PY if RAGAS_VENV_PY.exists() else Path(sys.executable)
    logf = open(RAGAS_LOGS / "evaluate.log", "w", encoding="utf-8")  # 每次平台启动清空, 日志只含本轮
    logf.write(f"[platform] {datetime.now():%Y-%m-%d %H:%M:%S} 由聚合平台启动, 模式: {', '.join(modes) or '全部'}\n")
    logf.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen(
        [str(py), "scripts/evaluate.py"] + modes,
        cwd=str(RAGAS_DIR), stdout=logf, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    # 记录本轮运行的模式(雷达图默认只画这些; 平台重启也保留)
    (RUNS_DIR / "last_eval.json").write_text(
        json.dumps({"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "modes": modes or sorted(discover_modes())}, ensure_ascii=False), encoding="utf-8")
    return {"started": True, "modes": modes or "all", "python": str(py)}


@app.get("/api/metrics-map")
def metrics_map():
    return METRIC_LABELS


@app.get("/api/metrics-dict")
def metrics_dict():
    """指标词典: metrics_dict.json 单一事实源 (概念/输入依赖/计算方法/方向/坑)."""
    p = Path(__file__).parent / "metrics_dict.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"metrics_dict.json 读取失败: {e}")


# ---- 公共检索参数 (所有检索模式共用; 改参数 = 需重新检索) ----
RETRIEVAL_DEFAULTS = {"top_k": 40, "chunk_top_k": 20, "max_total_tokens": 8000,
                      "enable_rerank": False, "response_type": "Multiple Paragraphs"}


def _retrieval_params() -> dict:
    rp = ((_load_module_cfg().get("rag") or {}).get("retrieval")) or {}
    out = dict(RETRIEVAL_DEFAULTS)
    for k in RETRIEVAL_DEFAULTS:
        if k in rp and rp[k] is not None and rp[k] != "":
            out[k] = rp[k]
    out["top_k"] = int(out["top_k"]); out["chunk_top_k"] = int(out["chunk_top_k"])
    out["max_total_tokens"] = int(out["max_total_tokens"])
    out["enable_rerank"] = bool(out["enable_rerank"])
    return out


def _param_fp() -> str:
    import hashlib
    s = ",".join(f"{k}={v}" for k, v in sorted(_retrieval_params().items()))
    qf = RAGAS_DATA / "eval_questions.jsonl"
    if qf.exists():
        s += "|qset:" + hashlib.md5(qf.read_bytes()).hexdigest()[:8]
    return hashlib.md5(s.encode()).hexdigest()[:8]


@app.get("/api/retrieval-params")
def get_retrieval_params():
    p = _retrieval_params()
    # 各模式数据文件的指纹(不一致 = 数据是旧参数的, 需重新检索)
    fps = {}
    for mode in discover_modes():
        meta = RAGAS_DATA / f"eval_{mode}.meta.json"
        try:
            fps[mode] = json.loads(meta.read_text(encoding="utf-8")).get("fp")
        except (OSError, json.JSONDecodeError):
            fps[mode] = None
    stale = [m for m, f in fps.items() if f != _param_fp()]
    return {"params": p, "fp": _param_fp(), "mode_fps": fps, "stale_modes": stale}


@app.post("/api/retrieval-params")
async def save_retrieval_params(req: Request):
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体错误")
    cfg = _load_module_cfg()
    rag = cfg.setdefault("rag", {})
    rp = rag.setdefault("retrieval", {})
    int_keys = ("top_k", "chunk_top_k", "max_total_tokens")
    for k in int_keys:
        if body.get(k) not in (None, ""):
            try:
                rp[k] = int(body[k])
            except (TypeError, ValueError):
                raise HTTPException(400, f"{k} 须为整数")
    if body.get("enable_rerank") is not None:
        rp["enable_rerank"] = bool(body["enable_rerank"])
    if body.get("response_type") in ("Multiple Paragraphs", "Single Paragraph"):
        rp["response_type"] = body["response_type"]
    MODULE_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "params": _retrieval_params(), "fp": _param_fp(),
            "note": "参数已变更, 受影响的模式需'重新检索'后再评测"}


@app.post("/api/retrieve")
def retrieve_run():
    """后台拉起 retrieve.py(6 模式×公共参数, .venv-lightrag 环境)生成评测数据."""
    import psutil
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
            if (len(cmd) >= 2
                    and str(cmd[0]).lower().rsplit("\\", 1)[-1] in ("python.exe", "python")
                    and str(cmd[1]).replace("\\", "/").endswith("retrieve.py")):
                raise HTTPException(409, "检索已在运行中")
        except HTTPException:
            raise
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if eval_running():
        raise HTTPException(409, "评测运行中, 请先停止评测再检索")
    py = RAGAS_DIR / ".venv-lightrag" / "Scripts" / "python.exe"
    if not py.exists():
        raise HTTPException(404, f"未找到 {py}")
    logf = open(RAGAS_LOGS / "retrieve.log", "w", encoding="utf-8")
    logf.write(f"[platform] {datetime.now():%Y-%m-%d %H:%M:%S} 由聚合平台启动检索, 参数: {_retrieval_params()}\n")
    logf.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen([str(py), "scripts/retrieve.py"], cwd=str(RAGAS_DIR),
                     stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)
    return {"started": True}


@app.get("/api/retrieve/status")
def retrieve_status():
    """检索进程状态 + 日志尾部."""
    import psutil
    running = False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info["cmdline"] or []
            if (len(cmd) >= 2
                    and str(cmd[0]).lower().rsplit("\\", 1)[-1] in ("python.exe", "python")
                    and str(cmd[1]).replace("\\", "/").endswith("retrieve.py")):
                running = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    p = RAGAS_LOGS / "retrieve.log"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-30:] if p.exists() else []
    return {"running": running, "lines": lines}


# evaluate.py build_metrics 的指标数(主流程 17 项; noise_sensitivity 由 fill_noise 专项补).
# 进度展示用, 若指标清单变化需同步.
EVAL_N_METRICS = 17


@app.get("/api/cache-status")
def cache_status():
    """各模式断点状态: cached 条数 vs total(题数×指标数), partial = 中断待续. 附最近一轮运行的模式.
    缓存 key 带检索参数指纹({mode}@{fp}:{qid}:{metric}), 参数变 -> 旧分自动不计入."""
    cache = load_cache()
    out = {}
    for mode in discover_modes():
        rows = read_jsonl(RAGAS_DATA / f"eval_{mode}.jsonl")
        total = len(rows) * EVAL_N_METRICS
        cfp = _mode_content_fp(mode)
        cached = sum(1 for k in cache if cfp and k.startswith(f"{mode}@{cfp}:"))
        state = "done" if total and cached >= total else ("partial" if cached else "empty")
        out[mode] = {"cached": cached, "total": total, "state": state}
    last = {}
    p = RUNS_DIR / "last_eval.json"
    if p.exists():
        try:
            last = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            last = {}
    return {"modes": out, "last_run": last, "param_fp": _param_fp()}


@app.post("/api/cache-clear")
async def cache_clear(req: Request):
    """删除指定模式的断点缓存 (scores_cache.json 里 '{mode}:' 前缀的条目), 原子重写."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体错误")
    modes = body.get("modes") or []
    if not isinstance(modes, list) or not modes:
        raise HTTPException(400, "modes 须为非空字符串数组")
    known = set(discover_modes())
    bad = [m for m in modes if m not in known]
    if bad:
        raise HTTPException(400, f"未知模式: {bad}; 可用: {sorted(known)}")
    if eval_running():
        raise HTTPException(409, "评测运行中, 不能清缓存")

    cache = load_cache()
    prefixes = tuple(f"{m}:" for m in modes) + tuple(f"{m}@" for m in modes)
    kept = {k: v for k, v in cache.items() if not k.startswith(prefixes)}
    removed = len(cache) - len(kept)
    p = RAGAS_RESULTS / "scores_cache.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return {"removed": removed, "remaining": len(kept), "modes": modes}


OLD_MODULES_PLACEHOLDER = None  # 模块状态改为按引擎安装情况动态计算 (见文件末尾 MODULES)


@app.get("/api/modules")
def modules():
    """四大引擎接入状态(按安装/配置情况动态驱动)。"""
    mods = [dict(m) for m in MODULES]
    for m in mods:
        if m["id"] == "langfuse" and _agent_up():
            # Langfuse 状态跟随智能体服务: 启动时配了 LANGFUSE_* 即点亮
            try:
                d = httpx.get(f"{AGENT_SVC}/health", timeout=3).json()
                m["ok"] = bool(d.get("langfuse"))
            except Exception:  # noqa: BLE001
                pass
    return mods


# ---------- 智能体服务(代理 + 启动) ----------
def _agent_up() -> bool:
    try:
        httpx.get(f"{AGENT_SVC}/health", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/agent/status")
def agent_status():
    if not _agent_up():
        return {"up": False, "langfuse": False}
    d = httpx.get(f"{AGENT_SVC}/health", timeout=3).json()
    return {"up": True, "langfuse": bool(d.get("langfuse"))}


@app.post("/api/agent/start")
def agent_start():
    """拉起智能体服务; 已在运行则先停止再拉起(重启语义, 用于应用新配置如 Langfuse key)。"""
    restart = False
    if _agent_up():
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(str(c) for c in (proc.info["cmdline"] or []))
                if "server.py" in cmd and str(AGENT_DIR) in cmd and proc.info["pid"] != os.getpid():
                    proc.terminate()
                    restart = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # 等端口释放
        import time
        for _ in range(20):
            if not _agent_up():
                break
            time.sleep(0.5)
    server_py = AGENT_DIR / "server.py"
    if not server_py.exists():
        raise HTTPException(404, f"未找到 {server_py}")
    # 智能体模块独立模型配置(key/url/AGENT_MODEL) + 平台基础环境 + langfuse
    env = _module_env("agent")
    for k, v in _read_env_pairs(LANGFUSE_ENV).items():
        if k.startswith("LANGFUSE"):
            env[k] = v
    logf = open(AGENT_DIR / "server.log", "a", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen(
        [str(AGENT_DIR / ".venv" / "Scripts" / "python.exe"), "server.py"],
        cwd=str(AGENT_DIR), stdout=logf, stderr=subprocess.STDOUT,
        env=env, creationflags=flags,
    )
    return {"started": True, "restarted": restart}


@app.api_route("/api/agent/proxy/{path:path}", methods=["GET", "POST"])
async def agent_proxy(path: str, request: Request):
    """转发到智能体服务 (chat/files/file/tasks/run_task)。"""
    body = await request.body()
    try:
        r = httpx.request(request.method, f"{AGENT_SVC}/{path}", content=body,
                          headers={"Content-Type": "application/json"}, timeout=300)
        return JSONResponse(r.json(), status_code=r.status_code)
    except httpx.ConnectError:
        raise HTTPException(502, "智能体服务未启动, 请先点『启动服务』")
    except json.JSONDecodeError:
        raise HTTPException(502, "智能体服务返回异常")


# ---------- M3: 智能体轨迹 + ragas agent 指标 ----------
AGENT_RUNS = AGENT_DIR / "runs"
AGENT_SCORES = RAGAS_RESULTS / "agent_scores.json"
AGENT_VENV_PY = AGENT_DIR / ".venv" / "Scripts" / "python.exe"
AGEN_PID = BASE_DIR / "runs" / "agent_generate.pid"
AGEV_PID = BASE_DIR / "runs" / "agent_eval.pid"


@app.get("/api/agent/trajs")
def agent_trajs():
    """轨迹文件列表 + 生成状态。"""
    files = sorted(AGENT_RUNS.glob("trajectories-*.jsonl"), reverse=True) if AGENT_RUNS.exists() else []
    return {"running": _tracked_running(AGEN_PID, "main.py"),
            "files": [{"name": p.name, "n": len(read_jsonl(p)),
                       "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")}
                      for p in files[:10]]}


@app.post("/api/agent/generate")
def agent_generate():
    """拉起智能体跑全部任务 → 生成轨迹 (agent 独立 venv + 模块模型 + langfuse 上报)。"""
    if _tracked_running(AGEN_PID, "main.py"):
        raise HTTPException(409, "轨迹生成已在运行中")
    if not AGENT_VENV_PY.exists():
        raise HTTPException(404, f"智能体环境不存在: {AGENT_VENV_PY}")
    env = _module_env("agent")
    for k, v in _read_env_pairs(LANGFUSE_ENV).items():
        if k.startswith("LANGFUSE"):
            env[k] = v
    _spawn([str(AGENT_VENV_PY), "main.py", "--all"], AGENT_DIR,
           RUNS_DIR / "agent_generate.log", env=env, pid_file=AGEN_PID)
    return {"started": True}


@app.post("/api/agent/eval")
def agent_eval():
    """对最新轨迹跑 ragas agent 指标 (ToolCallAccuracy / AgentGoalAccuracy)。"""
    if _tracked_running(AGEV_PID, "eval_agent"):
        raise HTTPException(409, "智能体指标评测已在运行中")
    if not (AGENT_RUNS.exists() and list(AGENT_RUNS.glob("trajectories-*.jsonl"))):
        raise HTTPException(404, "没有轨迹数据, 请先生成轨迹")
    env = _module_env("agent")
    env["JUDGE_MODEL"] = ((_load_module_cfg().get("agent") or {}).get("model")) or "gpt-4o-mini"
    _spawn([str(RAGAS_VENV_PY), "scripts/eval_agent.py"], RAGAS_DIR,
           RUNS_DIR / "agent_eval.log", env=env, pid_file=AGEV_PID)
    return {"started": True}


@app.get("/api/agent/scores")
def agent_scores():
    running = _tracked_running(AGEV_PID, "eval_agent")
    if not AGENT_SCORES.exists():
        return {"exists": False, "running": running}
    try:
        return {"exists": True, "running": running,
                **json.loads(AGENT_SCORES.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, OSError):
        return {"exists": False, "running": running}


# ---------- M2/M4 引擎: OpenCompass + promptfoo ----------
ENGINES_DIR = BASE_DIR / "engines"
RUNS_DIR = BASE_DIR / "runs"
OC_VENV_PY = BASE_DIR / ".venv-opencompass" / "Scripts" / "python.exe"
PROMPTFOO_CMD = BASE_DIR / "node_modules" / ".bin" / "promptfoo.cmd"


def _engine_env() -> dict:
    """平台独立 .env -> 子进程环境 (OPENAI_API_KEY 等)。"""
    env = {k: v for k, v in os.environ.items()}
    p = BASE_DIR / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _spawn(cmd: list, cwd: Path, log_path: Path, env: dict | None = None, pid_file: Path | None = None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")
    logf.write(f"\n[platform] {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
    logf.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT,
                     env=env or _engine_env(), creationflags=flags)
    if pid_file is not None:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(proc.pid))
    return proc


def _tracked_running(pid_file: Path, *keywords: str) -> bool:
    """pidfile 精确占用检测: PID 存活且命令行含全部关键词 (免疫无关进程的子串碰撞)。"""
    if not pid_file.exists():
        return False
    try:
        p = psutil.Process(int(pid_file.read_text().strip()))
        cmd = " ".join(str(c) for c in (p.cmdline() or []))
        return all(k in cmd for k in keywords) if keywords else True
    except (psutil.NoSuchProcess, ValueError, psutil.AccessDenied):
        pid_file.unlink(missing_ok=True)
        return False


def _engine_running(*keywords: str) -> bool:
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] == me:
                continue
            cmd = " ".join(str(c) for c in (proc.info["cmdline"] or []))
            # 多关键词须全部命中(如 main.py + --all), 避免调试命令误触发占用检测
            if cmd and all(k in cmd for k in keywords):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


# ---- M2: LLM 基准 (OpenCompass) ----
LLM_PID = BASE_DIR / "runs" / "llm" / "bench.pid"
LLM_STATE = BASE_DIR / "runs" / "llm" / "state.json"


def _foreign_run_bench() -> list[int]:
    """非平台登记(手动启动)的 run_bench 进程 PID —— 平台无法接管的占用者."""
    tracked = None
    if LLM_PID.exists():
        try:
            tracked = int(LLM_PID.read_text().strip())
        except ValueError:
            tracked = None
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.info["pid"] in (os.getpid(), tracked):
                continue
            parts = proc.info["cmdline"] or []
            if not parts:
                continue
            # 只认 python 解释器进程, 避免匹配到命令行恰好含相同字符串的 bash/IDE 进程
            exe = str(parts[0]).lower().rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            if exe not in ("python.exe", "python", "pythonw.exe"):
                continue
            cmd = " ".join(str(c) for c in parts)
            if "run_bench.py" in cmd and "opencompass" in cmd:
                found.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _llm_target(model: str | None = None) -> dict:
    """被测模型与端点解析: 模型登记(runs/llm/models.json) > 模块配置 > 平台 .env > 默认."""
    name = (model or "").strip()
    reg = next((m for m in _load_llm_models() if m.get("model") == name), None) if name else None
    c = _load_module_cfg().get("llm") or {}
    env = _engine_env()
    base_url = ((reg or {}).get("base_url") or c.get("base_url")
                or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if "\\" in base_url:
        raise HTTPException(400, f"base_url 含反斜杠(请求会变成 %5C 导致 404): {base_url}")
    return {"model": (name or c.get("model") or "gpt-4o-mini").strip(),
            "base_url": base_url,
            "api_key": (reg or {}).get("api_key") or c.get("api_key") or env.get("OPENAI_API_KEY", ""),
            "think": (reg or {}).get("think") or "default",
            "url": base_url.rstrip("/") + "/chat/completions"}


def _llm_preflight(model: str | None = None) -> dict:
    """单请求预检: 验证 模型名/密钥/端点. ok=True 可跑; 失败带 message."""
    t = _llm_target(model)
    if not t["api_key"]:
        return {"ok": False, "model": t["model"], "base_url": t["base_url"],
                "message": "✗ 未配置 API key (模块配置『⚙️ 模型设置』或平台 .env)"}
    import time
    t0 = time.time()
    try:
        r = httpx.post(t["url"],
                       json={"model": t["model"],
                             "messages": [{"role": "user", "content": "ping"}],
                             "max_tokens": 16},  # =1 会卡死 deepseek-v4-flash(服务端 bug)
                       headers={"Authorization": f"Bearer {t['api_key']}"}, timeout=20)
        ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            return {"ok": True, "model": t["model"], "base_url": t["base_url"],
                    "latency_ms": ms, "message": f"✓ 模型可达 ({ms}ms)"}
        hint = {401: "密钥认证失败", 403: "无权限(密钥/额度)", 404: "模型名不存在或端点路径不对",
                422: "模型名不存在", 429: "触发限流", 400: "请求格式被拒(检查模型名)"}.get(r.status_code, f"HTTP {r.status_code}")
        return {"ok": False, "status": r.status_code, "latency_ms": ms,
                "model": t["model"], "base_url": t["base_url"],
                "message": f"✗ {hint}: {r.text[:200]}"}
    except httpx.TimeoutException:
        return {"ok": False, "model": t["model"], "base_url": t["base_url"],
                "message": "✗ 连接超时(20s), 检查 base_url 与网络"}
    except httpx.HTTPError as e:
        return {"ok": False, "model": t["model"], "base_url": t["base_url"],
                "message": f"✗ 连接失败: {type(e).__name__}: {str(e)[:150]}"}


def _llm_progress(state: dict) -> dict | None:
    """本轮产出进度: 只认本轮启动后创建的实验目录.
    推理阶段数 predictions; 推理完成后进入评分阶段, 改数 results (避免推理满 52 后一直显示 100%).
    单子集基准(subjects=1, 如 AIME/GPQA)按题数统计: 在途 tmp_*.jsonl 行数 + 已交卷分片条目数 ——
    学科粒度下要等整个分片跑完才从 0 跳 100, 中途完全黑盒。"""
    model = (state or {}).get("model")
    if not model:
        return None
    binfo = LLM_BENCHES.get((state or {}).get("bench") or "") or {}
    total = binfo.get("subjects")
    try:
        t0 = datetime.strptime((state or {}).get("started") or "", "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        t0 = 0
    cands = [d for d in (RUNS_DIR / "llm" / "work").glob("*/")
             if d.is_dir() and d.stat().st_mtime >= t0 - 60]
    pred = res = 0
    cur = None
    if cands:
        cur = max(cands, key=lambda d: d.stat().st_mtime)
        for sub in ("predictions", "results"):
            mdir = cur / sub / model
            if mdir.is_dir():
                n = len(list(mdir.glob("*.json")))
                if sub == "predictions":
                    pred = n
                else:
                    res = n
    if total == 1 and binfo:  # 单子集: 学科粒度无意义, 换题级进度
        if res:
            return {"done": res, "total": 1, "pct": 100, "phase": "eval"}
        qtotal = max(binfo.get("est") or 1, 1)
        answered = 0
        if cur:
            pdir = cur / "predictions" / model
            if pdir.is_dir():
                for f in pdir.glob("tmp_*.jsonl"):
                    try:
                        answered += sum(1 for _ in f.open(encoding="utf-8"))
                    except OSError:
                        pass
                for f in pdir.glob("*.json"):
                    try:
                        d = json.loads(f.read_text(encoding="utf-8"))
                        answered += len(d) if isinstance(d, dict) else len(d or [])
                    except (json.JSONDecodeError, OSError):
                        pass
        return {"done": min(answered, qtotal), "total": qtotal,
                "pct": round(min(answered, qtotal) * 100 / qtotal), "phase": "infer"}
    if total and pred >= total:
        return {"done": res, "total": total, "pct": round(res * 100 / total), "phase": "eval"}
    return {"done": pred, "total": total,
            "pct": round(pred * 100 / total) if total else None, "phase": "infer"}


LLM_ERR_MARKERS = ("Request failed", "status code 40", "status code 42", "Traceback",
                   "Not Found", "Invalid", "未解析到", "ERROR")


def _llm_log_errors() -> list[str]:
    """日志尾部最近的可疑错误行(去重, 供状态栏警示)."""
    p = RUNS_DIR / "llm" / "log.txt"
    if not p.exists():
        return []
    found, seen = [], set()
    for line in reversed(p.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]):
        s = line.strip()
        if not s or s in seen or not any(m in s for m in LLM_ERR_MARKERS):
            continue
        seen.add(s)
        found.append(s[:220])
        if len(found) >= 3:
            break
    return found


def _llm_worker_errors() -> list[str]:
    """最新 work 目录下各 worker .out 里的 ERROR 行(429/401/404 等真实失败都记在这里, 主日志看不到)."""
    work = RUNS_DIR / "llm" / "work"
    dirs = sorted([d for d in work.glob("*/logs") if d.is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
    if not dirs:
        return []
    out_files = sorted((dirs[0] / "infer").glob("**/*.out"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    found, seen = [], set()
    for f in out_files[:16]:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            # .out 里 tqdm 用 \r 刷新进度条, 一行内可能夹带进度残留与 ANSI 转义; 先取最后一个片段,
            # 再兜底从 "OpenCompass - ERROR" 时间戳处截断, 保证只显示错误本体
            seg = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line.rsplit("\r", 1)[-1]).strip()
            m = re.search(r"\d{2}/\d{2} \d{2}:\d{2}:\d{2} - OpenCompass - ERROR", seg)
            if m:
                seg = seg[m.start():]
            if not seg or "ERROR" not in seg or seg in seen:
                continue
            seen.add(seg)
            found.append(seg[:220])
            if len(found) >= 5:
                return found
    return found


@app.post("/api/llm/check")
async def llm_check(req: Request):
    """预检被测模型(不启动任务): 单请求验证 模型名/密钥/端点, 跑基准前先点一下可避免空烧."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    return _llm_preflight((body.get("model") or "").strip() or None)


@app.post("/api/llm/stop")
def llm_stop():
    """停止基准: 强杀进程树 (run_bench→opencompass→workers); 手动启动的进程一并处理."""
    pids = _foreign_run_bench()
    if LLM_PID.exists():
        try:
            pids.append(int(LLM_PID.read_text().strip()))
        except ValueError:
            pass
    if not pids:
        raise HTTPException(404, "没有正在运行的基准")
    killed = []
    for pid in pids:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=30)
            else:
                psutil.Process(pid).kill()
            killed.append(pid)
        except (subprocess.TimeoutExpired, psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    LLM_PID.unlink(missing_ok=True)
    return {"stopped": True, "pids": killed}


@app.post("/api/llm/run")
async def llm_run(req: Request):
    if _tracked_running(LLM_PID, "run_bench"):
        raise HTTPException(409, "基准已在运行中")
    foreign = _foreign_run_bench()
    if foreign:
        raise HTTPException(409, f"检测到手动启动的基准进程 (PID: {foreign}), 平台无法接管, 请先停止")
    bench_script = ENGINES_DIR / "opencompass" / "run_bench.py"
    if not OC_VENV_PY.exists() or not bench_script.exists():
        raise HTTPException(404, "OpenCompass 未安装或缺少 run_bench.py")
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    # 模型优先级: 请求显式指定(模型管理一键运行) > 模块独立配置 > 默认
    mc = (_load_module_cfg().get("llm") or {}).get("model")
    model = ((body.get("model") or "").strip() or mc or "gpt-4o-mini")
    bench = (body.get("bench") or "ceval").strip()
    if bench not in LLM_BENCHES:
        raise HTTPException(400, f"未知基准 {bench}, 可选: {', '.join(LLM_BENCHES)}")
    # 并发数(前端填写): 传给 run_bench 的 OC_WORKERS (每 worker 1 qps × 1 并发调用)
    try:
        workers = int(body.get("workers") or 2)
    except (TypeError, ValueError):
        raise HTTPException(400, "workers 须为整数")
    if not 1 <= workers <= 16:
        raise HTTPException(400, "workers 须在 1-16 之间")
    # 预检(默认开): 模型名/密钥/端点不对就地报错, 不再空烧重试 (body {"check": false} 跳过)
    if body.get("check") is not False:
        pre = _llm_preflight(model)
        if not pre["ok"]:
            raise HTTPException(502, f"预检未通过(未启动任务): {pre['message']}")
    env = _engine_env()
    t = _llm_target(model)
    if t["api_key"]:
        env["OPENAI_API_KEY"] = t["api_key"]
    env["OPENAI_BASE_URL"] = t["base_url"]
    env["OC_WORKERS"] = str(workers)
    env["OC_THINK"] = t["think"]  # 思考强度(default/off/low/medium/high), run_bench 探测网关后内联
    _spawn([str(OC_VENV_PY), str(bench_script), model, bench], ENGINES_DIR / "opencompass",
           RUNS_DIR / "llm/log.txt", env=env, pid_file=LLM_PID)
    LLM_STATE.parent.mkdir(parents=True, exist_ok=True)
    LLM_STATE.write_text(json.dumps({
        "pid": LLM_PID.read_text().strip(), "model": model, "bench": bench,
        "label": LLM_BENCHES[bench]["label"], "workers": workers, "think": t["think"],
        "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False), encoding="utf-8")
    return {"started": True, "model": model, "bench": bench, "workers": workers, "think": t["think"]}


@app.get("/api/llm/status")
def llm_status():
    running = _tracked_running(LLM_PID, "run_bench")
    state = {}
    if LLM_STATE.exists():
        try:
            state = json.loads(LLM_STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
    errors = _llm_log_errors() + _llm_worker_errors()
    return {"running": running, "installed": OC_VENV_PY.exists(),
            "state": state, "progress": _llm_progress(state),
            "errors": list(dict.fromkeys(errors)), "foreign": _foreign_run_bench()}


@app.get("/api/llm/log")
def llm_log(n: int = 30):
    p = RUNS_DIR / "llm" / "log.txt"
    src = RUNS_DIR / "llm_bench.log"  # 手动跑时的日志
    q = p if p.exists() else src
    lines = []
    if q.exists():
        raw = q.read_text(encoding="utf-8", errors="replace").splitlines()
        # tqdm 在重定向日志里只留下冻结残影(0%| 0/2 之类), 无信息量且误导"卡住", 过滤掉
        keep = [s for s in raw if s.strip() and not re.search(r"\d+%\|", s)]
        lines = keep[-n:]
    return {"lines": lines, "worker_errors": _llm_worker_errors()}


@app.get("/api/llm/results")
def llm_results():
    """成绩矩阵: runs/llm/latest.json ({model: {dataset: {accuracy, n}}})"""
    p = RUNS_DIR / "llm" / "latest.json"
    if not p.exists():
        return {"exists": False}
    try:
        return {"exists": True, "matrix": json.loads(p.read_text(encoding="utf-8")),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")}
    except (json.JSONDecodeError, OSError):
        return {"exists": False}


@app.delete("/api/llm/results")
def llm_results_del(model: str):
    """删除某模型的全部测评结果: 成绩矩阵 + 运行历史 + 归档明细 (模型登记本身保留)。"""
    name = (model or "").strip()
    if not name:
        raise HTTPException(400, "model 不能为空")
    if _tracked_running(LLM_PID, "run_bench") or _foreign_run_bench():
        raise HTTPException(409, "基准运行中, 结束后再删(否则跑完合并会把成绩写回)")
    removed = {"matrix": 0, "history": 0, "archive": 0}
    p = RUNS_DIR / "llm" / "latest.json"
    if p.exists():
        try:
            matrix = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            matrix = {}
        if name in matrix:
            del matrix[name]
            removed["matrix"] = 1
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)
    if LLM_HISTORY.exists():
        kept, dropped = [], 0
        for line in LLM_HISTORY.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if h.get("model") == name:
                dropped += 1
            else:
                kept.append(line)
        if dropped:
            removed["history"] = dropped
            LLM_HISTORY.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    arch = RUNS_DIR / "llm" / "archive"
    if arch.is_dir():
        for ap in arch.glob("*.json"):
            try:
                if json.loads(ap.read_text(encoding="utf-8")).get("model") == name:
                    ap.unlink()
                    removed["archive"] += 1
            except (json.JSONDecodeError, OSError):
                continue
    if not any(removed.values()):
        raise HTTPException(404, f"没有 {name} 的测评结果")
    return {"deleted": name, **removed}


# ---- M2: 模型登记管理 + 运行历史 + 基准注册表 ----
LLM_MODELS_FILE = RUNS_DIR / "llm" / "models.json"
LLM_HISTORY = RUNS_DIR / "llm" / "history.jsonl"
# 与 engines/opencompass/run_bench.py 的 BENCHES 保持一致; subjects=结果子集数(进度条用)
LLM_BENCHES = {
    "ceval": {"label": "C-Eval", "note": "中文综合知识 · 52 学科", "est": 260, "subjects": 52},
    "cmmlu": {"label": "CMMLU", "note": "中文综合知识 · 67 学科", "est": 335, "subjects": 67},
    "mmlu": {"label": "MMLU", "note": "英文综合知识 · 57 学科", "est": 285, "subjects": 57},
    "gpqa": {"label": "GPQA", "note": "研究生级理化生 · diamond 全量 198 题", "est": 198, "subjects": 1},
    "aime2025": {"label": "AIME 2025", "note": "美国数学邀请赛 · 全量 30 题", "est": 30, "subjects": 1},
    "mmlu_pro": {"label": "MMLU-Pro", "note": "MMLU 继任 · 10 选项研究生级 · 每类前 20 题", "est": 280, "subjects": 14},
}


# ---- M2: 学科中文名映射(矩阵/雷达展示用) ----
LLM_SUBJECTS_CACHE: dict | None = None
BBH_ZH = {  # opencompass 无 BBH 中文映射, 手工对照官方任务名
    "boolean_expressions": "布尔表达式", "causal_judgement": "因果判断",
    "date_understanding": "日期理解", "disambiguation_qa": "指代消歧",
    "dyck_languages": "括号序列", "formal_fallacies": "形式谬误",
    "geometric_shapes": "几何图形", "hyperbaton": "语句重排",
    "logical_deduction_three_objects": "逻辑推演·3对象", "logical_deduction_five_objects": "逻辑推演·5对象",
    "logical_deduction_seven_objects": "逻辑推演·7对象", "movie_recommendation": "电影推荐",
    "multistep_arithmetic_two": "多步算术", "navigate": "空间导航",
    "object_counting": "物体计数", "penguins_in_a_table": "表格推理",
    "reasoning_about_colored_objects": "彩色物体推理", "ruin_names": "名称恶搞",
    "salient_translation_error_detection": "翻译纠错", "snarks": "讽刺识别",
    "sports_understanding": "体育常识", "temporal_sequences": "时间排序",
    "tracking_shuffled_objects_three_objects": "追踪打乱·3对象", "tracking_shuffled_objects_five_objects": "追踪打乱·5对象",
    "tracking_shuffled_objects_seven_objects": "追踪打乱·7对象", "web_of_lies": "谎言网络",
    "word_sorting": "单词排序",
}


@app.get("/api/llm/subjects")
def llm_subjects():
    """数据集 key(如 ceval-accountant) → 中文含义; ceval/cmmlu 从 opencompass 官方配置解析, BBH 手工对照."""
    global LLM_SUBJECTS_CACHE
    if LLM_SUBJECTS_CACHE is not None:
        return LLM_SUBJECTS_CACHE
    out = {f"bbh-{k}": v for k, v in BBH_ZH.items()}
    sp = (BASE_DIR / ".venv-opencompass" / "Lib" / "site-packages"
          / "opencompass" / "configs" / "datasets")
    try:  # ceval: 'accountant': ['Accountant', '注册会计师', 'STEM']
        text = (sp / "ceval" / "ceval_gen_5f30c7.py").read_text(encoding="utf-8")
        for m in re.finditer(r"'([a-z_]+)':\s*\['[^']*',\s*'([^']+)'(?:,\s*'([^']+)')?\]", text):
            out[f"ceval-{m.group(1)}"] = m.group(2) + (f" · {m.group(3)}" if m.group(3) else "")
    except OSError:
        pass
    try:  # cmmlu: 'agronomy': '农学'
        text = (sp / "cmmlu" / "cmmlu_gen_c13365.py").read_text(encoding="utf-8")
        for m in re.finditer(r"'([a-z_]+)':\s*'([^']+)'", text):
            out[f"cmmlu-{m.group(1)}"] = m.group(2)
    except OSError:
        pass
    LLM_SUBJECTS_CACHE = out
    return out


@app.get("/api/llm/benches")
def llm_benches():
    """可用基准注册表 (数据已本地化, 均为限题小集口径)."""
    return {"benches": [{"key": k, **v} for k, v in LLM_BENCHES.items()]}


def _load_llm_models() -> list[dict]:
    if LLM_MODELS_FILE.exists():
        try:
            return json.loads(LLM_MODELS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


@app.get("/api/llm/models")
def llm_models():
    """登记的模型列表(端点/思考强度可见, key 打码) + 已测过的模型(来自成绩矩阵)。"""
    matrix = {}
    p = RUNS_DIR / "llm" / "latest.json"
    if p.exists():
        try:
            matrix = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    out = [{"model": m.get("model"), "note": m.get("note", ""),
            "base_url": m.get("base_url", ""), "api_key_masked": _mask(m.get("api_key", "")),
            "think": m.get("think") or "default", "added": m.get("added", "")}
           for m in _load_llm_models()]
    return {"models": out, "tested": sorted(matrix.keys())}


THINK_LEVELS = ("default", "off", "low", "medium", "high")


@app.post("/api/llm/models")
async def llm_model_add(req: Request):
    """登记或更新被测模型(同名 = 更新, 非空字段覆盖, api_key 留空保持不变)。
    think 为通用思考强度五档: default/off/low/medium/high."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体错误")
    name = (body.get("model") or "").strip()
    if not name:
        raise HTTPException(400, "model 不能为空")
    think = (body.get("think") or "default").strip()
    if think not in THINK_LEVELS:
        raise HTTPException(400, f"think 须为 {'/'.join(THINK_LEVELS)}")
    base_url = (body.get("base_url") or "").strip()
    if "\\" in base_url:
        raise HTTPException(400, "base_url 不能含反斜杠(会导致 404)")
    models = _load_llm_models()
    cur = next((m for m in models if m.get("model") == name), None)
    if cur:  # 更新
        for k in ("note", "base_url", "think"):
            v = body.get(k)
            if isinstance(v, str) and v.strip():
                cur[k] = v.strip()
        if (body.get("api_key") or "").strip():
            cur["api_key"] = body["api_key"].strip()
        updated = True
    else:  # 新登记
        models.append({"model": name, "note": (body.get("note") or "").strip(),
                       "base_url": base_url, "api_key": (body.get("api_key") or "").strip(),
                       "think": think, "added": datetime.now().strftime("%m-%d %H:%M")})
        updated = False
    LLM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LLM_MODELS_FILE.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"total": len(models), "updated": updated}


@app.delete("/api/llm/models")
def llm_model_del(model: str):
    models = _load_llm_models()
    rest = [m for m in models if m["model"] != model]
    if len(rest) == len(models):
        raise HTTPException(404, f"没有登记名为 {model} 的模型")
    LLM_MODELS_FILE.write_text(json.dumps(rest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"deleted": model, "total": len(rest)}


@app.get("/api/llm/history")
def llm_history():
    """历次基准运行 (run_bench.py 逐次追加)。"""
    return {"items": read_jsonl(LLM_HISTORY)[-20:]}


# ---- M4: 红队扫描 (promptfoo redteam) ----
RT_CASES_FILE = ENGINES_DIR / "promptfoo" / "cases.json"
RT_SCAN_PID = BASE_DIR / "runs" / "redteam" / "scan.pid"


def _load_rt_cases() -> list[dict]:
    if RT_CASES_FILE.exists():
        try:
            return json.loads(RT_CASES_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


@app.get("/api/redteam/cases")
def rt_cases():
    """攻击用例库 (cases.json, 扫描时实时使用)."""
    return {"cases": _load_rt_cases()}


@app.post("/api/redteam/cases")
async def rt_case_add(req: Request):
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体错误")
    at, pr, rb = ((body.get("attack_type") or "").strip(),
                  (body.get("prompt") or "").strip(),
                  (body.get("rubric") or "").strip())
    if not (at and pr and rb):
        raise HTTPException(400, "attack_type / prompt / rubric 均必填")
    cases = _load_rt_cases()
    if any(c["attack_type"] == at for c in cases):
        raise HTTPException(409, f"攻击类型[{at}]已存在")
    cases.append({"attack_type": at, "prompt": pr, "rubric": rb})
    RT_CASES_FILE.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"total": len(cases)}


@app.delete("/api/redteam/cases")
def rt_case_del(attack_type: str):
    cases = _load_rt_cases()
    rest = [c for c in cases if c["attack_type"] != attack_type]
    if len(rest) == len(cases):
        raise HTTPException(404, f"没有名为[{attack_type}]的攻击类型")
    if not rest:
        raise HTTPException(400, "至少保留一条攻击用例")
    RT_CASES_FILE.write_text(json.dumps(rest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"deleted": attack_type, "total": len(rest)}


@app.post("/api/redteam/scan")
def redteam_scan():
    if _tracked_running(RT_SCAN_PID, "promptfoo"):
        raise HTTPException(409, "promptfoo 任务已在运行中")
    if not PROMPTFOO_CMD.exists():
        raise HTTPException(404, "promptfoo 未安装")
    if not _agent_up():
        raise HTTPException(502, "靶机(智能体服务)未启动, 请先在智能体页签启动服务")
    cases = _load_rt_cases()
    if not cases:
        raise HTTPException(404, "没有攻击用例 (engines/promptfoo/cases.json)")
    (RUNS_DIR / "redteam").mkdir(exist_ok=True)
    (RUNS_DIR / "redteam" / ".archived").unlink(missing_ok=True)  # 新扫描: 重置懒归档标记
    (RUNS_DIR / "redteam" / "result.json").unlink(missing_ok=True)
    # 裁判模型取红队模块独立配置; 用例来自 cases.json, 动态生成 yaml (key/url 走环境变量注入)
    judge = ((_load_module_cfg().get("redteam") or {}).get("model")) or "gpt-4o-mini"

    def y(s) -> str:  # JSON 字符串即合法 YAML 双引号标量, 转义安全
        return json.dumps(str(s), ensure_ascii=False)

    tpl = ["# 自动生成: cases.json → dyn yaml (勿手改)",
           "description: " + y(f"红队扫描 · 四大名著智能体 ({len(cases)} 用例)"),
           "targets:", "  - id: http", "    config:",
           "      url: " + y(f"{AGENT_SVC}/chat"),
           "      method: POST", "      headers:", "        Content-Type: application/json",
           "      body:", "        message: \"{{prompt}}\"",
           "      transformResponse: json.reply",
           "defaultTest:", "  options:", f"    provider: openai:{judge}",
           "  assert:", "    - type: llm-rubric", "      value: \"{{rubric}}\"", "tests:"]
    for c in cases:
        tpl += ["  - vars:", "      attack_type: " + y(c["attack_type"]),
                "      prompt: " + y(c["prompt"]), "      rubric: " + y(c["rubric"])]
    dyn_yaml = RUNS_DIR / "redteam" / "dyn_redteam.yaml"
    dyn_yaml.write_text("\n".join(tpl) + "\n", encoding="utf-8")
    _spawn(["cmd", "/c", str(PROMPTFOO_CMD), "eval",
            "-c", str(dyn_yaml),
            "--output", str(RUNS_DIR / "redteam" / "result.json")],
           BASE_DIR, RUNS_DIR / "redteam" / "log.txt", env=_module_env("redteam"),
           pid_file=RT_SCAN_PID)
    return {"started": True, "judge": judge, "cases": len(cases)}


@app.get("/api/redteam/results")
def redteam_results():
    p = RUNS_DIR / "redteam" / "result.json"
    if not p.exists():
        return {"exists": False, "running": _tracked_running(RT_SCAN_PID, "promptfoo")}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d.get("results", {}).get("results", [])
        findings, passes = [], 0
        for r in rows:
            meta = r.get("metadata") or {}
            rvars = r.get("vars") or {}
            raw_prompt = r.get("prompt")
            if isinstance(raw_prompt, dict):
                raw_prompt = raw_prompt.get("raw") or ""
            raw_prompt = str(raw_prompt or rvars.get("prompt", ""))
            resp = r.get("response")
            reply = (resp.get("output") or "") if isinstance(resp, dict) else str(resp or "")
            item = {
                "pass": bool(r.get("success")),
                "plugin": meta.get("plugin") or rvars.get("attack_type", "未分类"),
                "strategy": meta.get("strategy", "basic"),
                "prompt": raw_prompt[:160],
                "reply": str(reply)[:200],
            }
            findings.append(item)
            passes += item["pass"]
        running = _tracked_running(RT_SCAN_PID, "promptfoo")
        # 扫描完成且尚未归档 → 懒归档进"历史轮次"(下次扫描前完成落账)
        if not running:
            _redteam_lazy_archive(p, len(rows), passes, findings)
        return {"exists": True, "running": running,
                "total": len(rows), "passed": passes, "failed": len(rows) - passes,
                "findings": findings,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")}
    except (json.JSONDecodeError, OSError):
        return {"exists": False, "running": _tracked_running(RT_SCAN_PID, "promptfoo")}


def _redteam_lazy_archive(p: Path, total: int, passes: int, findings: list):
    """result.json 首次被读取(扫描已结束)时归档一份; .archived 标记防止重复。"""
    marker = p.parent / ".archived"
    if marker.exists():
        return
    try:
        judge = ((_load_module_cfg().get("redteam") or {}).get("model")) or "gpt-4o-mini"
        arch_dir = p.parent / "archive"
        arch_dir.mkdir(exist_ok=True)
        entry = {
            "ts": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "judge": judge, "model": judge, "total": total, "passed": passes,
            "pass_rate": round(passes / total, 3) if total else None,
            "findings": findings,
        }
        (arch_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")).write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        marker.touch()
    except OSError:
        pass


MODULES = [
    {"id": "ragas", "name": "ragas", "dim": "RAG + 智能体指标", "ms": "M1", "ok": True},
    {"id": "opencompass", "name": "OpenCompass", "dim": "模型能力基准", "ms": "M2", "ok": OC_VENV_PY.exists()},
    {"id": "langfuse", "name": "Langfuse", "dim": "轨迹采集 · 生产监控", "ms": "M3", "ok": False},
    {"id": "promptfoo", "name": "promptfoo", "dim": "安全红队", "ms": "M4", "ok": PROMPTFOO_CMD.exists()},
]


# ---------- 统一历史轮次 (四模块: rag/llm/agent/redteam 归档目录的浏览与删除) ----------
HISTORY_DIRS = {
    "rag": RAGAS_DIR / "results" / "archive",          # 子目录: meta.json + eval_report.md + scores_*
    "llm": RUNS_DIR / "llm" / "archive",               # json: {ts, model, bench, n, mean, scores}
    "agent": RAGAS_DIR / "results" / "agent_archive",  # json: 完整 agent_scores 内容
    "redteam": RUNS_DIR / "redteam" / "archive",       # json: {ts, judge, total, passed, findings}
}
RAG_CORE = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")


def _rag_history_item(d: Path) -> dict:
    meta = {}
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    modes = meta.get("modes") or []
    # 汇总分: 各模式核心5指标均值再取平均
    vals, n_questions = [], meta.get("n") or {}
    for m in modes:
        try:
            s = json.loads((d / f"scores_{m}.json").read_text(encoding="utf-8"))
            vals.extend(v for k in RAG_CORE if isinstance(s.get(k), (int, float)) for v in [s[k]])
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "id": d.name, "ts": meta.get("ts") or d.name[:15],
        "title": "+".join(modes) if modes else d.name,
        "chips": {"裁判": meta.get("judge", "?"), "参数指纹": meta.get("param_fp", "旧格式"),
                  "embedding": meta.get("embedding", "-"),
                  "题数": ",".join(f"{k}:{v}" for k, v in n_questions.items()) or "-"},
        "summary": f"核心指标均值 {round(sum(vals) / len(vals), 3)}" if vals else "无分数(中断轮)",
        "has_report": (d / "eval_report.md").exists(),
    }


def _json_history_item(module: str, p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"id": p.stem, "ts": p.stem, "title": p.stem, "chips": {}, "summary": "(损坏)", "has_report": False}
    if module == "llm":
        return {"id": p.stem, "ts": d.get("ts", p.stem), "title": f"{d.get('model')} × {d.get('bench')}",
                "chips": {"被测模型": d.get("model", "?"), "基准": d.get("bench", "?"), "子集数": d.get("n", "?")},
                "summary": f"accuracy {d.get('mean')}", "has_report": True}
    if module == "agent":
        s = d.get("summary") or {}
        return {"id": p.stem, "ts": d.get("ts", p.stem), "title": f"智能体 {d.get('n', '?')} 任务",
                "chips": {"裁判": d.get("judge", "?"), "轨迹": str(d.get("source", "?"))[:34]},
                "summary": f"TCA {s.get('tool_call_accuracy', '-')} · Goal {s.get('goal_accuracy', '-')} · 召回 {s.get('tool_recall', '-')}",
                "has_report": True}
    if module == "redteam":
        return {"id": p.stem, "ts": d.get("ts", p.stem), "title": f"红队 {d.get('total', '?')} 用例",
                "chips": {"裁判": d.get("judge", "?")},
                "summary": f"通过率 {d.get('pass_rate', '-')} ({d.get('passed', '-')}/{d.get('total', '-')})",
                "has_report": True}
    return {"id": p.stem, "ts": p.stem, "title": p.stem, "chips": {}, "summary": "", "has_report": False}


@app.get("/api/history/{module}")
def history_list(module: str):
    if module not in HISTORY_DIRS:
        raise HTTPException(404, f"未知模块: {module}")
    root = HISTORY_DIRS[module]
    items = []
    if root.exists():
        for child in root.iterdir():
            if module == "rag":
                if child.is_dir() and not child.name.startswith("."):
                    items.append(_rag_history_item(child))
            elif child.is_file() and child.suffix == ".json":
                items.append(_json_history_item(module, child))
    items.sort(key=lambda x: x["id"], reverse=True)
    return {"module": module, "items": items}


@app.get("/api/history/{module}/{rid}")
def history_detail(module: str, rid: str):
    if module not in HISTORY_DIRS:
        raise HTTPException(404, f"未知模块: {module}")
    if "/" in rid or "\\" in rid or ".." in rid:
        raise HTTPException(400, "非法 id")
    root = HISTORY_DIRS[module] / rid
    if module == "rag":
        rep = root / "eval_report.md"
        if not root.is_dir() or not rep.exists():
            raise HTTPException(404, "该轮无报告")
        return {"kind": "markdown", "content": rep.read_text(encoding="utf-8", errors="replace")[:60000]}
    root = HISTORY_DIRS[module] / f"{rid}.json"  # 列表项 id 是不带后缀的 stem
    if not root.is_file():
        raise HTTPException(404, "记录不存在")
    return {"kind": "json", "content": json.dumps(
        json.loads(root.read_text(encoding="utf-8")), ensure_ascii=False, indent=1)[:60000]}


@app.delete("/api/history/{module}/{rid}")
def history_delete(module: str, rid: str):
    if module not in HISTORY_DIRS:
        raise HTTPException(404, f"未知模块: {module}")
    if "/" in rid or "\\" in rid or ".." in rid:
        raise HTTPException(400, "非法 id")
    import shutil
    root = HISTORY_DIRS[module] / rid
    if module == "rag":
        if root.is_dir():
            shutil.rmtree(root)
            return {"deleted": True}
    else:
        root = HISTORY_DIRS[module] / f"{rid}.json"  # 列表项 id 是不带后缀的 stem
        if root.is_file():
            root.unlink()
            return {"deleted": True}
    raise HTTPException(404, "记录不存在")


# ---------- Langfuse 轨迹代理 (云端数据, 平台内直接查看) ----------
def _langfuse_auth() -> tuple[str, str] | None:
    lf = _read_env_pairs(LANGFUSE_ENV)
    pk, sk = lf.get("LANGFUSE_PUBLIC_KEY", ""), lf.get("LANGFUSE_SECRET_KEY", "")
    return (pk, sk) if pk and sk else None


@app.get("/api/traces")
def traces_list(limit: int = 15, page: int = 1):
    """最近轨迹列表 (代理 Langfuse /api/public/traces)。"""
    auth = _langfuse_auth()
    if not auth:
        return {"configured": False}
    import base64
    lf = _read_env_pairs(LANGFUSE_ENV)
    host = (lf.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    cred = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    try:
        r = httpx.get(f"{host}/api/public/traces",
                      params={"limit": limit, "page": page},
                      headers={"Authorization": f"Basic {cred}"}, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        items = []
        for t in data:
            usage = t.get("usage") or {}
            items.append({
                "id": t.get("id"), "name": t.get("name"),
                "timestamp": (t.get("timestamp") or "")[:19].replace("T", " "),
                "latency_s": round(t["latency"], 2) if t.get("latency") is not None else None,
                "total_tokens": usage.get("total") or 0,
                "observations": len(t.get("observations") or []),
            })
        return {"configured": True, "items": items,
                "host": host, "page": page}
    except Exception as e:  # noqa: BLE001
        return {"configured": True, "error": f"{type(e).__name__}: {str(e)[:120]}"}


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str):
    """单条轨迹详情 (含 observations 层级与 input/output)。"""
    auth = _langfuse_auth()
    if not auth:
        return {"configured": False}
    import base64
    lf = _read_env_pairs(LANGFUSE_ENV)
    host = (lf.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    cred = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    try:
        r = httpx.get(f"{host}/api/public/traces/{trace_id}",
                      headers={"Authorization": f"Basic {cred}"}, timeout=15)
        r.raise_for_status()
        t = r.json()
        obs = []
        for o in (t.get("observations") or []):
            usage = o.get("usage") or {}
            obs.append({
                "type": o.get("type"), "name": o.get("name"),
                "latency_s": round(o["latency"], 2) if o.get("latency") is not None else None,
                "model": o.get("model"),
                "total_tokens": usage.get("total") or 0,
                "input": str(o.get("input") or "")[:600],
                "output": str(o.get("output") or "")[:600],
            })
        return {"configured": True, "trace": {
            "id": t.get("id"), "name": t.get("name"),
            "timestamp": (t.get("timestamp") or "")[:19].replace("T", " "),
            "input": str(t.get("input") or "")[:400],
            "output": str(t.get("output") or "")[:800],
            "observations": obs,
        }}
    except Exception as e:  # noqa: BLE001
        return {"configured": True, "error": f"{type(e).__name__}: {str(e)[:120]}"}


# ---------- 设置 (配置读写: 平台 .env + ragas/langfuse/langfuse.env) ----------
OPENAI_ENV = BASE_DIR / ".env"
LANGFUSE_ENV = RAGAS_DIR / "langfuse" / "langfuse.env"
LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
OPENAI_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")


def _read_env_pairs(path: Path) -> dict:
    """读 env 文件里未注释的键值对。"""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def _mask(v: str) -> str:
    return f"{v[:8]}...{v[-4:]}" if v and len(v) > 16 else ("已设置" if v else "")


@app.get("/api/settings")
def get_settings():
    oai = _read_env_pairs(OPENAI_ENV)
    lf = _read_env_pairs(LANGFUSE_ENV)
    lf = {k: lf.get(k, "") for k in LANGFUSE_KEYS}
    # langfuse 是否已配置 = 两个 key 都非空且非占位
    lf_ready = bool(lf["LANGFUSE_PUBLIC_KEY"] and lf["LANGFUSE_SECRET_KEY"])
    return {
        "openai": {"api_key_masked": _mask(oai.get("OPENAI_API_KEY", "")),
                   "base_url": oai.get("OPENAI_BASE_URL", ""),
                   "configured": bool(oai.get("OPENAI_API_KEY"))},
        "langfuse": {"public_key_masked": _mask(lf["LANGFUSE_PUBLIC_KEY"]),
                     "secret_key_masked": _mask(lf["LANGFUSE_SECRET_KEY"]),
                     "host": lf["LANGFUSE_HOST"] or "https://cloud.langfuse.com",
                     "configured": lf_ready,
                     "agent_running": _agent_up()},
    }


def _upsert_env_file(path: Path, updates: dict):
    """更新 env 文件: 原位替换已有键, 没有则追加; 保留注释与其他行。"""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = set()
    out = []
    for line in lines:
        s = line.strip()
        replaced = False
        for k, v in updates.items():
            if s.startswith(k + "="):
                out.append(f"{k}={v}")
                done.add(k)
                replaced = True
                break
        if not replaced:
            out.append(line)
    for k, v in updates.items():
        if k not in done:
            out.append(f"{k}={v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


@app.post("/api/settings")
async def save_settings(req: Request):
    """只更新请求里非空字段; 空字段保持原值。保存后平台进程环境同步刷新。"""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    oai_updates, lf_updates = {}, {}
    if body.get("openai_api_key"):
        oai_updates["OPENAI_API_KEY"] = body["openai_api_key"].strip()
    if body.get("openai_base_url"):
        oai_updates["OPENAI_BASE_URL"] = body["openai_base_url"].strip()
    if body.get("langfuse_public_key"):
        lf_updates["LANGFUSE_PUBLIC_KEY"] = body["langfuse_public_key"].strip()
    if body.get("langfuse_secret_key"):
        lf_updates["LANGFUSE_SECRET_KEY"] = body["langfuse_secret_key"].strip()
    if body.get("langfuse_host"):
        lf_updates["LANGFUSE_HOST"] = body["langfuse_host"].strip()

    msg = []
    if oai_updates:
        _upsert_env_file(OPENAI_ENV, oai_updates)
        msg.append("OpenAI 配置已保存")
    if lf_updates:
        _upsert_env_file(LANGFUSE_ENV, lf_updates)
        msg.append("Langfuse 配置已保存")
    if not msg:
        return {"saved": False, "message": "没有需要保存的字段"}
    return {"saved": True, "message": "；".join(msg) + "。Langfuse 需重启智能体服务后生效"}


@app.post("/api/settings/test-langfuse")
def test_langfuse():
    """用当前配置 ping Langfuse 服务端健康接口 (pk:sk 做 basic auth)。"""
    lf = _read_env_pairs(LANGFUSE_ENV)
    pk, sk = lf.get("LANGFUSE_PUBLIC_KEY", ""), lf.get("LANGFUSE_SECRET_KEY", "")
    host = (lf.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    if not pk or not sk:
        return {"ok": False, "message": "public/secret key 未配置"}
    try:
        import base64
        auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        r = httpx.get(f"{host}/api/public/health",
                      headers={"Authorization": f"Basic {auth}"}, timeout=10)
        if r.status_code == 200:
            return {"ok": True, "message": f"连接成功 ({host})"}
        if r.status_code in (401, 403):
            return {"ok": False, "message": f"服务可达但 key 认证失败 (HTTP {r.status_code})"}
        return {"ok": False, "message": f"HTTP {r.status_code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"连不上 {host}: {type(e).__name__}"}


# ---------- RAG 题集管理 (reference 构建: 查看/AI合成/采纳) ----------
QUESTIONS_FILE = RAGAS_DATA / "eval_questions.jsonl"
SYNTH_PENDING = RAGAS_DATA / "questions_synth.jsonl"
SYNTH_PID = BASE_DIR / "runs" / "rag_synth.pid"
RAGAS_VENV_PY = Path(r"D:\job\.venv\Scripts\python.exe")  # ragas 环境
NEEDS_REFERENCE = {"context_precision", "context_recall", "context_entity_recall",
                   "factual_correctness", "answer_correctness", "semantic_similarity",
                   "noise_sensitivity", "rougeL", "bleu", "chrf",
                   "string_similarity", "exact_match"}


@app.get("/api/rag/questions")
def rag_questions():
    """当前题集 + 指标的 reference 依赖标注."""
    return {"questions": read_jsonl(QUESTIONS_FILE), "needs_reference": sorted(NEEDS_REFERENCE)}


@app.post("/api/rag/questions")
async def rag_question_add(req: Request):
    """手动添加一题 (问题 + reference 要点 + 可选黄金段落)."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体错误")
    q, ref = (body.get("user_input") or "").strip(), (body.get("reference") or "").strip()
    if not q or not ref:
        raise HTTPException(400, "user_input 和 reference 必填")
    cur = read_jsonl(QUESTIONS_FILE)
    new_id = max((int(x.get("id", 0)) for x in cur), default=0) + 1
    row = {"id": new_id, "book": (body.get("book") or "未知").strip(),
           "user_input": q, "reference": ref}
    ctx = [c.strip() for c in (body.get("reference_contexts") or []) if str(c).strip()]
    if ctx:
        row["reference_contexts"] = ctx
    cur.append(row)
    QUESTIONS_FILE.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in cur) + "\n", encoding="utf-8")
    return {"id": new_id, "total": len(cur)}


@app.delete("/api/rag/questions")
def rag_question_delete(id: int):
    cur = read_jsonl(QUESTIONS_FILE)
    rest = [x for x in cur if int(x.get("id", 0)) != id]
    if len(rest) == len(cur):
        raise HTTPException(404, f"没有 id={id} 的题")
    QUESTIONS_FILE.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rest) + "\n", encoding="utf-8")
    return {"deleted": id, "total": len(rest)}


@app.post("/api/rag/synth")
async def rag_synth(req: Request):
    """AI 合成题集 (ragas TestsetGenerator), 产出待审核文件."""
    if _tracked_running(SYNTH_PID, "synth_questions"):
        raise HTTPException(409, "合成任务已在运行中")
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    size = int(body.get("size") or 10)
    script = RAGAS_DIR / "scripts" / "synth_questions.py"
    if not script.exists():
        raise HTTPException(404, f"未找到 {script}")
    SYNTH_PENDING.unlink(missing_ok=True)
    _spawn([str(RAGAS_VENV_PY), str(script), str(size)], RAGAS_DIR,
           RUNS_DIR / "rag_synth.log", env=_engine_env(), pid_file=SYNTH_PID)
    return {"started": True, "size": size}


@app.get("/api/rag/synth")
def rag_synth_status():
    return {"running": _tracked_running(SYNTH_PID, "synth_questions"),
            "pending": read_jsonl(SYNTH_PENDING)}


@app.post("/api/rag/synth/adopt")
async def rag_synth_adopt(req: Request):
    """采纳待审题为正式题 (ids 为空 = 全部采纳); 采纳后从待审区移除."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    ids = set(body.get("ids") or [])
    pending = read_jsonl(SYNTH_PENDING)
    if not pending:
        raise HTTPException(404, "没有待审核的合成题")
    adopted = [p for p in pending if not ids or p.get("id") in ids]
    if not adopted:
        raise HTTPException(400, "没有匹配的题")
    cur = read_jsonl(QUESTIONS_FILE)
    next_id = max((int(q.get("id", 0)) for q in cur), default=0)
    for p in adopted:
        next_id += 1
        cur.append({"id": next_id, "book": p.get("book", "未知"),
                    "user_input": p["user_input"], "reference": p["reference"],
                    **({"reference_contexts": p["reference_contexts"]}
                       if p.get("reference_contexts") else {})})
    QUESTIONS_FILE.write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in cur) + "\n", encoding="utf-8")
    rest = [p for p in pending if p not in adopted]
    SYNTH_PENDING.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in rest) + ("\n" if rest else ""),
        encoding="utf-8")
    return {"adopted": len(adopted), "total_questions": len(cur)}


@app.post("/api/rag/synth/discard")
async def rag_synth_discard(req: Request):
    """丢弃待审题 (ids 为空 = 清空)."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    ids = set(body.get("ids") or [])
    pending = read_jsonl(SYNTH_PENDING)
    rest = [p for p in pending if ids and p.get("id") not in ids] if ids else []
    SYNTH_PENDING.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in rest) + ("\n" if rest else ""),
        encoding="utf-8")
    return {"discarded": len(pending) - len(rest), "pending": len(rest)}


# ---------- 模块独立模型配置 (LLM基准 / 智能体 / 红队裁判) ----------
MODULE_CFG_PATH = BASE_DIR / "module_config.json"


def _load_module_cfg() -> dict:
    if MODULE_CFG_PATH.exists():
        try:
            return json.loads(MODULE_CFG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


@app.get("/api/module-config")
def get_module_config():
    """各模块模型配置; 未配置的字段运行时回退到平台 .env。rag 的 model=生成模型, judge_model=裁判模型."""
    cfg = _load_module_cfg()
    oai = _read_env_pairs(OPENAI_ENV)
    fallback = {"model": "gpt-4o-mini",
                "base_url": oai.get("OPENAI_BASE_URL", ""),
                "api_key_masked": _mask(oai.get("OPENAI_API_KEY", ""))}
    out = {}
    for m in ("llm", "agent", "redteam", "rag"):
        c = cfg.get(m) or {}
        item = {"model": c.get("model", ""), "base_url": c.get("base_url", ""),
                "api_key_masked": _mask(c.get("api_key", "")),
                "configured": bool(c.get("model") or c.get("base_url") or c.get("api_key")
                                   or c.get("judge_model")),
                "fallback": fallback}
        if m == "rag":
            item["judge_model"] = c.get("judge_model", "")
            # embedding/rerank 独立角色 (未配置时脚本回落 .env / 索引同源默认)
            item["embedding_model"] = c.get("embedding_model", "")
            item["embedding_base_url"] = c.get("embedding_base_url", "")
            item["embedding_api_key_masked"] = _mask(c.get("embedding_api_key", ""))
            item["rerank_model"] = c.get("rerank_model", "")
            item["rerank_base_url"] = c.get("rerank_base_url", "")
            item["rerank_api_key_masked"] = _mask(c.get("rerank_api_key", ""))
            item["configured"] = item["configured"] or bool(
                c.get("embedding_model") or c.get("embedding_base_url") or c.get("embedding_api_key")
                or c.get("rerank_model") or c.get("rerank_base_url") or c.get("rerank_api_key"))
        out[m] = item
    return out


@app.post("/api/module-config")
async def save_module_config(req: Request):
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "请求体错误")
    m = body.get("module")
    if m not in ("llm", "agent", "redteam", "rag"):
        raise HTTPException(400, "module 须为 llm/agent/redteam/rag")
    cfg = _load_module_cfg()
    c = cfg.setdefault(m, {})
    keys = ("model", "base_url", "api_key", "judge_model")
    if m == "rag":
        keys += ("embedding_model", "embedding_base_url", "embedding_api_key",
                 "rerank_model", "rerank_base_url", "rerank_api_key")
    for k in keys:
        v = (body.get(k) or "").strip()
        if v:
            c[k] = v
    MODULE_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    note = {"agent": "重启智能体服务(智能体页签『启动服务』)后生效",
            "rag": "下次检索(retrieve)与评测(evaluate)运行时生效"}.get(m, "下次运行时生效")
    return {"saved": True, "message": f"已保存。{note}"}


def _module_env(module: str) -> dict:
    """平台基础环境 + 该模块独立覆盖的 key/url/model。"""
    env = _engine_env()
    c = _load_module_cfg().get(module) or {}
    if c.get("api_key"):
        env["OPENAI_API_KEY"] = c["api_key"]
    if c.get("base_url"):
        env["OPENAI_BASE_URL"] = c["base_url"]
    if module == "agent" and c.get("model"):
        env["AGENT_MODEL"] = c["model"]
    return env


# ---------- M5: 评测历史与趋势 (跨模块汇总) ----------
@app.get("/api/history")
def history():
    """各模块评测历史 + RAG 综合分趋势数据。"""
    items = []
    # RAG (evaluate.py 每轮追加)
    rag_hist = read_jsonl(RAGAS_RESULTS / "history.jsonl")
    for h in rag_hist[-20:]:
        items.append({"module": "RAG", "ts": h.get("ts"), "mode": h.get("mode"),
                      "detail": f"{h.get('n')} 题 · 综合分 {h.get('overall')}",
                      "sort": h.get("ts")})
    # LLM 基准 (run_bench.py 每轮追加)
    for h in read_jsonl(LLM_HISTORY)[-20:]:
        items.append({"module": "LLM 基准", "ts": h.get("ts"),
                      "detail": f"{h.get('model')} · {h.get('bench')} · {h.get('n')} 项 均分 {(h.get('mean') or 0) * 100:.1f}",
                      "sort": h.get("ts")})
    # 红队
    p = RUNS_DIR / "redteam" / "result.json"
    if p.exists():
        ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        items.append({"module": "安全红队", "ts": ts,
                      "detail": "promptfoo 扫描 (明细见红队页签)", "sort": ts})
    # 智能体指标
    if AGENT_SCORES.exists():
        try:
            d = json.loads(AGENT_SCORES.read_text(encoding="utf-8"))
            s = d.get("summary") or {}
            items.append({"module": "智能体指标", "ts": d.get("ts"),
                          "detail": f"tca {s.get('tool_call_accuracy')} · goal {s.get('goal_accuracy')} · 工具召回 {s.get('tool_recall')}",
                          "sort": d.get("ts")})
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    items = [i for i in items if i.get("sort")]
    items.sort(key=lambda x: x["sort"], reverse=True)
    # 趋势: RAG 各模式综合分曲线
    trend: dict = {}
    for h in rag_hist:
        if h.get("mode") and h.get("overall") is not None:
            trend.setdefault(h["mode"], []).append({"ts": h["ts"], "overall": h["overall"]})
    return {"items": items[:30], "trend": trend}


# ---------- 静态页面 ----------
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    # no-cache: 每次都向服务器校验最新版, 避免改动后浏览器继续用旧页面 (CDN/新功能不同步踩过多次)
    return FileResponse(str(BASE_DIR / "static" / "index.html"),
                        headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    print(f"评测聚合平台: http://127.0.0.1:{PORT}  (聚合目录: {RAGAS_DIR})")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
