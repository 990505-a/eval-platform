# -*- coding: utf-8 -*-
"""任务跑批: 智能体服务逐题执行固定任务集 → 生成轨迹 trajectories-*.jsonl

被测对象: agents/mingzhu-agent (deepagents, :8820)。任务集由智能体服务自带(tasks.jsonl),
本脚本只做编排与证据采集 —— 每题执行后抓取工作区文件列表与长期记忆快照,
作为 evaluate.py 做终态核验(file_hit / memory_hit)的环境证据。

产出契约 (与平台 app.py / evaluate.py 对齐):
    runs/agenteval/trajectories-<ts>.jsonl
    每行 {id, instruction, ts, pred_tools, tool_calls, reply, latency_s,
          files, memory, expect:{tools, answer_contains, file, memory_contains}}

环境变量(平台注入; 单独手跑有默认值): AGENT_SVC_URL
自测: python run_tasks.py --selftest  (不调智能体, 写一份合成轨迹供链路验证)
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "agenteval"
AGENT = (os.environ.get("AGENT_SVC_URL") or "http://127.0.0.1:8820").rstrip("/")
MEMORY_FILE = "memory/long_term_memory.md"


def _get(path: str):
    r = httpx.get(f"{AGENT}{path}", timeout=600)
    r.raise_for_status()
    return r.json()


def _run_one(task: dict) -> dict:
    r = httpx.post(f"{AGENT}/run_task", json={"task_id": task["id"]}, timeout=600)
    r.raise_for_status()
    d = r.json()
    if d.get("error"):
        raise RuntimeError(f"任务{task['id']} 执行失败: {d['error']}")
    files = []
    try:
        files = [f["name"] for f in _get("/files")]
    except Exception as e:  # noqa: BLE001
        print(f"[run_tasks] 采集工作区失败: {e}", flush=True)
    memory = ""
    try:
        memory = _get(f"/file?name={MEMORY_FILE}").get("content", "")
    except Exception:  # noqa: BLE001  # 记忆文件尚不存在 = 没写过, 不是错误
        pass
    return {
        "id": task["id"], "instruction": task["instruction"],
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pred_tools": [c["name"] for c in d["tool_calls"]],
        "tool_calls": d["tool_calls"], "reply": d["reply"],
        "latency_s": d["latency_s"], "files": files, "memory": memory[-1500:],
        "expect": {"tools": task.get("expect_tools") or [],
                   "answer_contains": task.get("expect_answer_contains") or [],
                   "file": task.get("expect_file"),
                   "memory_contains": task.get("expect_memory_contains")},
    }


def selftest():
    """合成两条轨迹(不调智能体), 供评分链路与平台展示的离线验证。"""
    fake = [
        {"id": 1, "instruction": "(自测)检索贾宝玉原文并引用",
         "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "pred_tools": ["search_classics"],
         "tool_calls": [{"name": "search_classics", "args": "{\"query\": \"贾宝玉\"}",
                         "output": "…原文段落…"}],
         "reply": "贾宝玉是《红楼梦》主人公, 原文有云:'这个妹妹我曾见过的'。",
         "latency_s": 3.2, "files": [], "memory": "",
         "expect": {"tools": ["search_classics"], "answer_contains": ["贾宝玉"],
                    "file": None, "memory_contains": None}},
        {"id": 2, "instruction": "(自测)写入桃园结义报告",
         "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "pred_tools": ["search_classics", "write_file", "remember"],
         "tool_calls": [{"name": "search_classics", "args": "{}", "output": "…"},
                        {"name": "write_file", "args": "{\"path\": \"桃园结义报告.md\"}", "output": "已写入"},
                        {"name": "remember", "args": "{\"content\": \"用户偏好简洁\"}", "output": "已记住"}],
         "reply": "桃园结义报告已完成并写入文件。",
         "latency_s": 8.5, "files": ["桃园结义报告.md", "memory/long_term_memory.md"],
         "memory": "- 用户偏好简洁回答",
         "expect": {"tools": ["search_classics", "write_file"], "answer_contains": ["桃园"],
                    "file": "桃园结义报告.md", "memory_contains": "简洁"}},
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
    h = _get("/health")
    if not h.get("up"):
        raise SystemExit("智能体服务未启动")
    tasks = _get("/tasks")
    if not tasks:
        raise SystemExit("任务集为空: agents/mingzhu-agent/tasks.jsonl")
    print(f"[run_tasks] 智能体 {AGENT} · {len(tasks)} 题 · 模型 {h.get('model')}", flush=True)
    rows = []
    for i, t in enumerate(tasks, 1):
        print(f"[run_tasks] ({i}/{len(tasks)}) 任务{t['id']} 执行中… "
              f"({t['instruction'][:30]})", flush=True)
        t0 = time.time()
        try:
            rows.append(_run_one(t))
            print(f"[run_tasks]   完成 {time.time()-t0:.0f}s · 工具 "
                  f"{rows[-1]['pred_tools']}", flush=True)
        except Exception as e:  # noqa: BLE001  # 单题失败不中断整轮
            print(f"[run_tasks]   失败: {e}", flush=True)
    if rows:
        _write(rows)


if __name__ == "__main__":
    main()
