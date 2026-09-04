# -*- coding: utf-8 -*-
"""桩智能体: 不调 LLM, 演示平台接入契约(POST /run /health /files /file), 供链路验证.

用法:  python scripts/stub_agent.py          # 默认 :8820 (平台默认地址)
       set AGENT_PORT=8821 && python ...     # 换端口
体验:  起来后到平台「🤖智能体测评 → ②评测」点「🔍 检测接入」→ 任务集「▶ 试跑」→「🚀 一键评测」
换成真智能体: 参考 平台「② 讲解 · 接入指南」的参考实现 B.
"""
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

PORT = int(os.environ.get("AGENT_PORT", "8820"))
WS = Path(__file__).parent / "stub_workspace"
WS.mkdir(exist_ok=True)
TOOLS = ["web_search", "write_file", "read_file", "save_memory"]

app = FastAPI(title="stub-agent (接入契约演示)")


@app.get("/health")
def health():
    return {"up": True, "langfuse": False, "tools": TOOLS}


@app.post("/run")
def run(req: dict):
    t0 = time.time()
    ins = str(req.get("instruction") or "")
    calls = []
    if "搜索" in ins or "检索" in ins:
        calls.append({"name": "web_search", "args": {"query": ins[:30]}, "output": "(桩)检索到相关资料…"})
    if "长期记忆" in ins:
        (WS / "memory").mkdir(exist_ok=True)
        with (WS / "memory/long_term_memory.md").open("a", encoding="utf-8") as f:
            f.write("- 用户偏好简洁回答\n")
        calls.append({"name": "save_memory", "args": {"content": "简洁"}, "output": "已记住"})
    m = None
    for key in ("写入工作区文件", "写入文件", "整理后写入"):
        if key in ins:
            m = key
            break
    if m:
        tail = ins.split(m)[-1].strip().rstrip("。 ")
        fname = tail if tail.endswith(".md") else (tail + ".md" if tail else "报告.md")
        (WS / fname).write_text(f"(桩)关于 {ins[:40]} 的调查结论…", encoding="utf-8")
        calls.append({"name": "write_file", "args": {"path": fname}, "output": f"已写入 {fname}"})
    reply = f"(桩)已完成指令: {ins[:40]}…" if calls else f"(桩)收到指令: {ins[:40]}, 桩未定义对应工具行为"
    return {"reply": reply, "tool_calls": calls, "latency_s": round(time.time() - t0, 1)}


@app.post("/chat")
def chat(req: dict):
    return run(req)


@app.get("/files")
def files():
    return [{"name": str(p.relative_to(WS)), "size": p.stat().st_size, "mtime": ""}
            for p in sorted(WS.rglob("*")) if p.is_file()]


@app.get("/file")
def file_content(name: str):
    p = (WS / name).resolve()
    if not str(p).startswith(str(WS.resolve())):
        return {"content": ""}
    return {"content": p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""}


if __name__ == "__main__":
    print(f"桩智能体: http://127.0.0.1:{PORT}  (workspace={WS})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
