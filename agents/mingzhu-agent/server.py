# -*- coding: utf-8 -*-
"""名著智能体 —— deepagents 框架实现, 供平台智能体测评/红队扫描/对话测试的靶机服务.

契约 (与 eval-platform 前端/引擎对齐):
    GET  /health             → {"up", "langfuse", "rag", "model", "base_url"}
    POST /chat               → {"thread_id?", "message"} ⇒ {"thread_id", "reply",
                               "tool_calls": [{name, args}], "latency_s"}   (同线程共享历史)
    GET  /tasks              → [{id, instruction, expect_tools, ...}]      (固定任务集)
    POST /run_task           → {"task_id"} ⇒ {reply, tool_calls, tool_missing, latency_s}
    GET  /files              → [{name, size, mtime}]                       (工作区)
    GET  /file?name=         → {"content"}

能力: deepagents 规划 + 四件套工具 —— search_classics(名著 RAG 检索) / write_file / read_file /
remember(长期记忆)。任务与轨迹供 engines/deepeval 的 run_tasks.py 消费(线B: 环境真实副作用)。
Langfuse: 配置 LANGFUSE_* 时自动挂 callback 上报轨迹(本地或云端)。
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
PORT = int(os.environ.get("AGENT_PORT", "8820"))
WORKSPACE = APP_DIR / "workspace"
MEMORY_FILE = "memory/long_term_memory.md"
TASKS_FILE = APP_DIR / "tasks.jsonl"
# 平台 ⚙️ 模型设置(module_config.json 的 agent 段), compose 挂载进来; Docker/本地模式统一吃这份
MODULE_CFG = APP_DIR / "module_config.json"
MAX_THREAD_MESSAGES = 30  # 线程历史上限, 防长对话 token 失控


def _agent_cfg() -> tuple[str, str, str]:
    """模型配置: ⚙️ 模块设置优先, 未配置的字段回退 .env 环境变量(AGENT_MODEL/OPENAI_*)。"""
    model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    try:
        d = (json.loads(MODULE_CFG.read_text(encoding="utf-8")).get("agent") or {})
        model = d.get("model") or model
        base = (d.get("base_url") or base).rstrip("/")
        key = d.get("api_key") or key
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return model, base, key
RAG_URL = os.environ.get("RAG_URL", "http://rag-mingzhu:9621").rstrip("/")
RAG_MODE = os.environ.get("RAG_MODE", "mix")  # naive/local/global/hybrid/mix

LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
LANGFUSE_READY = all(os.environ.get(k) for k in LANGFUSE_KEYS) and not (
    os.environ.get("LANGFUSE_PUBLIC_KEY", "").startswith("pk-lf-xxxx"))

app = FastAPI(title="名著智能体 (deepagents)")

_agent = None  # 懒加载: 服务先起来, 模型配置问题留到请求时报可读错误
_threads: dict[str, list] = {}  # thread_id → 消息历史(进程内)


def _safe_path(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        raise HTTPException(400, f"路径越界: {rel}")
    return p


def _build_agent():
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from deepagents import create_deep_agent

    @tool
    def search_classics(query: str) -> str:
        """检索四大名著与莎士比亚作品的原文内容。输入应为具体的人物/情节/台词检索问题,
        返回相关原文段落与知识图谱上下文。回答名著相关问题前必须先调用本工具。"""
        r = httpx.post(f"{RAG_URL}/query", json={"query": query, "mode": RAG_MODE},
                       timeout=300)
        r.raise_for_status()
        return str(r.json().get("data") or r.json())[:12000]

    @tool
    def write_file(path: str, content: str) -> str:
        """把内容写入工作区文件(path 为相对路径, 如 报告.md)。产出交付物时使用。"""
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已写入 {path} ({len(content)} 字)"

    @tool
    def read_file(path: str) -> str:
        """读取工作区文件内容(相对路径)。"""
        p = _safe_path(path)
        if not p.exists():
            return f"文件不存在: {path}"
        return p.read_text(encoding="utf-8")[:8000]

    @tool
    def remember(content: str) -> str:
        """把重要信息写入长期记忆(跨对话持久)。用户表达偏好/约定时使用。"""
        p = _safe_path(MEMORY_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"- {datetime.now():%Y-%m-%d %H:%M} {content}\n")
        return f"已记住: {content}"

    system_prompt = (
        "你是名著研究员智能体, 精通中国四大名著(红楼梦/西游记/水浒传/三国演义)与莎士比亚全集。"
        "回答名著相关问题前, 必须先调用 search_classics 检索原文依据再作答, 并注明出处作品;"
        "用户要求保存结论/报告时用 write_file 落盘; 用户表达偏好或约定时用 remember 记入长期记忆;"
        "闲聊或与名著无关的问题直接回答。使用中文交流。"
    )
    model, base_url, api_key = _agent_cfg()
    print(f"[agent] 模型: {model} @ {base_url}", flush=True)
    # Kimi coding 系模型只允许 temperature=1, 不传让 SDK 用模型默认值最稳
    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=1)
    try:  # deepagents>=0.7 用 system_prompt; 兼容旧版 instructions
        return create_deep_agent(model=llm,
                                 tools=[search_classics, write_file, read_file, remember],
                                 system_prompt=system_prompt)
    except TypeError:
        return create_deep_agent(model=llm,
                                 tools=[search_classics, write_file, read_file, remember],
                                 instructions=system_prompt)


def _get_agent():
    global _agent
    if _agent is None:
        _, _, key = _agent_cfg()
        if not key or key.startswith("sk-xxx"):
            raise HTTPException(503, "未配置 OPENAI_API_KEY: 请编辑项目根目录 .env 后重启 "
                                     "(docker compose restart agent-mingzhu)")
        _agent = _build_agent()
    return _agent


def _callbacks():
    """Langfuse 轨迹上报 (未配置则不上报)。"""
    if not LANGFUSE_READY:
        return None
    try:
        from langfuse import CallbackHandler  # langfuse v3 SDK
        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        try:
            from langfuse.langchain import CallbackHandler as Cb2
            return [Cb2()]
        except Exception:  # noqa: BLE001
            return None


def _invoke(messages: list) -> tuple[str, list[dict], list]:
    """跑一轮智能体, 返回 (最终回复, 工具调用明细, 完整消息序列)。"""
    agent = _get_agent()
    cfg = {}
    cbs = _callbacks()
    if cbs:
        cfg["callbacks"] = cbs
    result = agent.invoke({"messages": messages}, cfg or None)
    final_msgs = result["messages"]
    reply = str(final_msgs[-1].content)
    tool_calls, outputs = [], []
    for m in final_msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_calls.append({"name": tc.get("name", ""),
                               "args": _clip(json.dumps(tc.get("args", {}), ensure_ascii=False), 300)})
        if m.__class__.__name__ == "ToolMessage" and getattr(m, "name", None):
            outputs.append(_clip(str(m.content), 200))
    for tc, out in zip(tool_calls, outputs):  # 数量一致时按序配对出输出预览
        tc["output"] = out
    return reply, tool_calls, final_msgs


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _rag_up() -> bool:
    try:
        httpx.get(f"{RAG_URL}/health", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_tasks() -> list[dict]:
    if not TASKS_FILE.exists():
        return []
    out = []
    for line in TASKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class ChatReq(BaseModel):
    thread_id: str = ""
    message: str


class RunTaskReq(BaseModel):
    task_id: int


@app.get("/health")
def health():
    model, base_url, _ = _agent_cfg()
    return {"up": True, "langfuse": LANGFUSE_READY, "rag": _rag_up(),
            "model": model, "base_url": base_url,
            "tasks": len(_load_tasks())}


@app.post("/chat")
def chat(req: ChatReq):
    from langchain_core.messages import HumanMessage
    tid = req.thread_id or f"t{int(time.time())}"
    try:
        history = _threads.setdefault(tid, [])
        history.append(HumanMessage(content=req.message))
        t0 = time.time()
        reply, tool_calls, final_msgs = _invoke(history)
        # 用智能体返回的完整消息序列做线程记忆(含 AI/工具消息), 截尾防 token 失控
        _threads[tid] = final_msgs[-MAX_THREAD_MESSAGES:]
        return {"thread_id": tid, "reply": reply, "tool_calls": tool_calls,
                "latency_s": round(time.time() - t0, 1)}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=502, content={"reply": "", "error": f"{type(e).__name__}: {e}"})


@app.get("/tasks")
def tasks():
    return _load_tasks()


@app.post("/run_task")
async def run_task(request: Request):
    body = await request.json()
    tid = int(body.get("task_id", 0))
    task = next((t for t in _load_tasks() if t.get("id") == tid), None)
    if not task:
        raise HTTPException(404, f"任务不存在: {tid}")
    from langchain_core.messages import HumanMessage
    try:
        t0 = time.time()
        reply, tool_calls, _ = _invoke([HumanMessage(content=task["instruction"])])
        called = [c["name"] for c in tool_calls]
        missing = [t for t in (task.get("expect_tools") or []) if t not in called]
        return {"task_id": tid, "reply": reply, "tool_calls": tool_calls,
                "tool_missing": missing, "latency_s": round(time.time() - t0, 1)}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=502, content={"reply": "", "error": f"{type(e).__name__}: {e}"})


@app.get("/files")
def files():
    if not WORKSPACE.exists():
        return []
    out = []
    for p in sorted(WORKSPACE.rglob("*")):
        if p.is_file():
            out.append({"name": str(p.relative_to(WORKSPACE)),
                        "size": p.stat().st_size,
                        "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M")})
    return out


@app.get("/file")
def file_content(name: str):
    p = _safe_path(name)
    if not p.exists():
        raise HTTPException(404, f"文件不存在: {name}")
    return {"content": p.read_text(encoding="utf-8", errors="replace")[:20000]}


if __name__ == "__main__":
    WORKSPACE.mkdir(exist_ok=True)
    m, b, _ = _agent_cfg()
    print(f"名著智能体: http://0.0.0.0:{PORT}  (model={m} @ {b}, rag={RAG_URL}, "
          f"tasks={len(_load_tasks())}, langfuse={'on' if LANGFUSE_READY else 'off'}, "
          f"workspace={WORKSPACE})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
