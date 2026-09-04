# -*- coding: utf-8 -*-
"""任务跑批: 逐题调用被测智能体 POST /run → 轨迹 + 环境终态证据 → trajectories-*.jsonl

被测对象: 你自己的智能体服务(任意框架, 接入契约见平台「智能体测评 → 接入指南」):
    POST /run        {"instruction": "..."} ⇒ {"reply", "tool_calls":[{name,args,output}], "latency_s"}
    GET  /health     ⇒ {"up": true, ...}                        (必需)
    GET  /files      ⇒ [{"name",...}]                           (可选: expect_file 终态核验)
    GET  /file?name= ⇒ {"content"}                              (可选: expect_file_contains 终态核验)

任务集: 平台侧管理(AGENT_TASKS_FILE, 默认 engines/deepeval/tasks/default.jsonl),
每行 {id, instruction, expect_tools, expect_answer_contains, expect_file, expect_file_contains}.

本脚本只做编排与证据采集 —— 每题执行后抓取工作区文件列表与指定文件内容,
作为 evaluate.py 做终态核验(file_hit / content_hit)的环境证据。

产出契约 (与平台 app.py / evaluate.py 对齐):
    runs/agenteval/trajectories-<ts>.jsonl
    每行 {id, instruction, ts, pred_tools, tool_calls, reply, latency_s, error,
          files, file_evidence, expect:{tools, answer_contains, file, file_contains}}

环境变量(平台注入; 单独手跑有默认值): AGENT_SVC_URL · AGENT_TASKS_FILE
自测: python run_tasks.py --selftest  (不调智能体, 写一份合成轨迹供链路验证)
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).parent
RUNS = HERE.parent.parent / "runs" / "agenteval"
AGENT = (os.environ.get("AGENT_SVC_URL") or "http://127.0.0.1:8820").rstrip("/")
TASKS_FILE = Path(os.environ.get("AGENT_TASKS_FILE") or HERE / "tasks" / "default.jsonl")


def _get(path: str, timeout: int = 30):
    r = httpx.get(f"{AGENT}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_tasks() -> list[dict]:
    if not TASKS_FILE.exists():
        raise SystemExit(f"任务集不存在: {TASKS_FILE} (平台「智能体测评→任务集」页可维护)")
    tasks = []
    for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            tasks.append(json.loads(line))
    return tasks


def _expect_of(task: dict) -> dict:
    fc = task.get("expect_file_contains")
    if isinstance(fc, str) and "::" in fc:          # 宽容格式 "文件::文本"
        f, _, t = fc.partition("::")
        fc = {"file": f.strip(), "text": t.strip()}
    return {"tools": task.get("expect_tools") or [],
            "answer_contains": task.get("expect_answer_contains") or [],
            "file": task.get("expect_file"),
            "file_contains": fc or None,
            # 旧任务集兼容: expect_memory_contains → 记忆文件内容核验
            "memory_contains": task.get("expect_memory_contains")}


def _run_one(task: dict) -> dict:
    row = {"id": task["id"], "instruction": task["instruction"],
           "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "pred_tools": [], "tool_calls": [], "reply": "", "latency_s": None,
           "error": None, "files": [], "file_evidence": {},
           "expect": _expect_of(task)}
    t0 = time.time()
    try:
        r = httpx.post(f"{AGENT}/run", json={"instruction": task["instruction"]}, timeout=600)
        r.raise_for_status()
        d = r.json()
        row["reply"] = str(d.get("reply") or "")
        row["tool_calls"] = d.get("tool_calls") or []
        row["pred_tools"] = [c.get("name", "") for c in row["tool_calls"]]
        row["latency_s"] = d.get("latency_s") or round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001  # 单题失败不中断整轮
        row["error"] = f"{type(e).__name__}: {e}"
        row["latency_s"] = round(time.time() - t0, 1)
        return row
    # ---- 环境终态证据: 文件列表 + 指定文件内容(供 file_hit / content_hit) ----
    try:
        row["files"] = [f["name"] for f in _get("/files")]
    except Exception as e:  # noqa: BLE001
        print(f"[run_tasks] 采集工作区失败(视为无文件证据): {e}", flush=True)
    want = []
    if row["expect"]["file_contains"]:
        want.append(row["expect"]["file_contains"]["file"])
    if row["expect"]["memory_contains"]:            # 旧字段: 默认记忆文件路径
        want.append("memory/long_term_memory.md")
    for path in want:
        try:
            row["file_evidence"][path] = _get(f"/file?name={path}").get("content", "")[-1500:]
        except Exception:  # noqa: BLE001  # 文件不存在 = 没写过, 不是错误
            row["file_evidence"][path] = ""
    return row


def selftest():
    """合成两条轨迹(不调智能体), 供评分链路与平台展示的离线验证。"""
    fake = [
        {"id": 1, "instruction": "(自测)检索'量子计算'的资料并引用一句原文。",
         "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "pred_tools": ["web_search"],
         "tool_calls": [{"name": "web_search", "args": "{\"query\": \"量子计算\"}",
                         "output": "…资料段落…"}],
         "reply": "量子计算利用叠加与纠缠并行处理信息, 资料有云:'量子比特可同时处于0和1'。",
         "latency_s": 3.2, "error": None, "files": [], "file_evidence": {},
         "expect": {"tools": ["web_search"], "answer_contains": ["量子"],
                    "file": None, "file_contains": None, "memory_contains": None}},
        {"id": 2, "instruction": "(自测)调查'可再生能源'并写入工作区文件 可再生能源报告.md。",
         "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "pred_tools": ["web_search", "write_file", "save_memory"],
         "tool_calls": [{"name": "web_search", "args": "{}", "output": "…"},
                        {"name": "write_file", "args": "{\"path\": \"可再生能源报告.md\"}", "output": "已写入"},
                        {"name": "save_memory", "args": "{\"content\": \"用户偏好简洁\"}", "output": "已记住"}],
         "reply": "可再生能源报告已完成并写入文件。",
         "latency_s": 8.5, "error": None,
         "files": ["可再生能源报告.md", "memory/long_term_memory.md"],
         "file_evidence": {"memory/long_term_memory.md": "- 用户偏好简洁"},
         "expect": {"tools": ["web_search", "write_file"], "answer_contains": ["可再生"],
                    "file": "可再生能源报告.md", "file_contains": None, "memory_contains": "简洁"}},
    ]
    _write(fake)
    print(f"[run_tasks] selftest 完成: {RUNS}/trajectories-*.jsonl (2 条合成轨迹)", flush=True)


def _write(rows: list):
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"trajectories-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 只保留最新 10 份轨迹
    for old in sorted(RUNS.glob("trajectories-*.jsonl"))[:-10]:
        old.unlink()
    print(f"[run_tasks] 轨迹落盘: {out.name} ({len(rows)} 题)", flush=True)
    return out


def main():
    if "--selftest" in sys.argv:
        return selftest()
    h = _get("/health", timeout=10)
    if not h.get("up", True):
        raise SystemExit("智能体服务 /health 返回未就绪")
    tasks = load_tasks()
    if not tasks:
        raise SystemExit(f"任务集为空: {TASKS_FILE}")
    print(f"[run_tasks] 智能体 {AGENT} · {len(tasks)} 题 · 任务集 {TASKS_FILE.name}", flush=True)
    rows = []
    for i, t in enumerate(tasks, 1):
        print(f"[run_tasks] ({i}/{len(tasks)}) 任务{t['id']} 执行中… "
              f"({t['instruction'][:30]})", flush=True)
        t0 = time.time()
        row = _run_one(t)
        rows.append(row)
        tag = f"失败:{row['error'][:60]}" if row["error"] else f"工具 {row['pred_tools']}"
        print(f"[run_tasks]   完成 {time.time()-t0:.0f}s · {tag}", flush=True)
    if rows:
        _write(rows)


if __name__ == "__main__":
    main()
