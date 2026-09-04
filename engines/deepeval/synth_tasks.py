# -*- coding: utf-8 -*-
"""智能体任务合成: 工具清单+领域素材(task_seeds.md) + 现有任务示例 → LLM 生成任务草稿 → 待审区.

用法 (平台经 sys.executable 拉起, 只依赖 httpx):
    python synth_tasks.py [N]        # 默认 5, clamp 到 [1, 20]
    python synth_tasks.py --selftest # 零 API 自测: 写 2 条硬编码任务到待审区

约定 (与 ragas 的 synth_questions.py 同构):
- 待审区 runs/agenteval/tasks_synth.jsonl, 追加式 + id 续号 + 原子写
- 每行: {id, instruction, expect_tools, expect_answer_contains, expect_file,
         expect_file_contains:{file,text}, rationale}    # expect_* 字段名是全链契约
- 采纳后由平台写入正式任务集 engines/deepeval/tasks/default.jsonl (AGENT_TASKS_FILE 可覆盖)
- expect_tools 只保留 task_seeds.md 工具清单里登记过的名字(非法工具名的任务直接丢弃)
- 进度一律 print("[synth-tasks] ...", flush=True), 日志落 runs/agent_tasksynth.log
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parent.parent          # 平台根
SEEDS = HERE / "task_seeds.md"
RUNS = ROOT / "runs" / "agenteval"
PENDING = RUNS / "tasks_synth.jsonl"
TASKS_FILE = Path(os.environ.get("AGENT_TASKS_FILE")
                  or HERE / "tasks" / "default.jsonl")

SELFTEST = "--selftest" in sys.argv


def load_env_file():
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _env(k: str, d: str) -> str:
    return (os.environ.get(k) or d).rstrip("/")


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("//"):
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def parse_seeds() -> tuple[list[str], str, str]:
    """task_seeds.md → (合法工具名列表, 工具清单原文, 领域素材原文)."""
    text = SEEDS.read_text(encoding="utf-8") if SEEDS.exists() else ""
    m_tools = re.search(r"##\s*工具清单\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL)
    m_dom = re.search(r"##\s*领域素材\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL)
    tools_txt = (m_tools.group(1) if m_tools else "").strip()
    domain = (m_dom.group(1) if m_dom else "").strip() or "(未提供, 按现有任务示例的风格出题)"
    tools = [mm.group(1) for mm in re.finditer(r"^-\s*([A-Za-z_][\w]*)\s*:", tools_txt, re.MULTILINE)]
    return tools, tools_txt or "(空)", domain[:3000]


def few_shot() -> str:
    tasks = read_jsonl(TASKS_FILE)[:3]
    if not tasks:
        return "(暂无示例任务)"
    keys = ("instruction", "expect_tools", "expect_answer_contains",
            "expect_file", "expect_file_contains")
    return "\n".join(json.dumps({k: t.get(k) for k in keys if t.get(k)},
                                ensure_ascii=False) for t in tasks)


PROMPT = """你要为一个智能体生成评测任务。该智能体的可用工具、领域背景和现有任务示例如下。

【工具清单】
{tools}

【领域素材】
{domain}

【现有任务示例(风格参照)】
{fewshot}

请生成 {n} 个新任务, 要求:
1. instruction 是发给智能体的中文指令, 必须能用上述工具完成, 不要提及"工具"字样
2. expect_tools 只能从工具清单的名字里选(数组, 1-3 个, 按必要顺序); 指令需要什么就标什么
3. 尽量带可判定的终态锚点: expect_answer_contains(回答必含的关键词数组, 2-4 个短词)、
   expect_file(指令要求产出的文件名, 仅当指令明确要求写文件时)、
   expect_file_contains(对象 {{"file": 路径, "text": 该文件须包含的关键词}}, 仅当指令要求
   把特定内容写入某个文件/记忆时)——没有把握的字段直接省略, 不要硬凑
4. 任务类型多样化: 单检索型 / 检索+写文件组合型 / 写入记忆型 / 读文件综合型
5. rationale 一句话说明这道题考什么(审核用)
6. 只输出 JSON 数组, 不要解释: [{{"instruction": "...", "expect_tools": ["..."], "expect_answer_contains": ["..."], "expect_file": "...", "expect_file_contains": {{"file": "...", "text": "..."}}, "rationale": "..."}}]"""


def _norm_task(t: dict, valid_tools: list[str]) -> dict | None:
    """字段清洗 + expect_tools 合法性过滤; 指令或工具为空的丢弃."""
    ins = str(t.get("instruction") or "").strip()
    tools = [str(x).strip() for x in (t.get("expect_tools") or [])
             if str(x).strip() in valid_tools]
    if not ins or not tools:
        return None
    out = {"instruction": ins, "expect_tools": tools}
    kws = [str(x).strip() for x in (t.get("expect_answer_contains") or []) if str(x).strip()]
    if kws:
        out["expect_answer_contains"] = kws[:4]
    v = str(t.get("expect_file") or "").strip()
    if v:
        out["expect_file"] = v
    fc = t.get("expect_file_contains")
    if isinstance(fc, dict) and fc.get("file") and fc.get("text"):
        out["expect_file_contains"] = {"file": str(fc["file"]).strip(),
                                       "text": str(fc["text"]).strip()}
    if str(t.get("rationale") or "").strip():
        out["rationale"] = str(t["rationale"]).strip()[:120]
    return out


def llm_generate(n: int) -> list[dict]:
    key = os.environ.get("OPENAI_API_KEY", "")
    base = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("AGENT_MODEL") or "gpt-4o-mini"
    if not key:
        sys.exit("[synth-tasks] 缺 OPENAI_API_KEY (平台 .env 或 ⚙️ 设置/agent 模块)")
    valid_tools, tools_txt, domain = parse_seeds()
    if not valid_tools:
        sys.exit(f"[synth-tasks] 种子文件未解析到工具清单(## 工具清单 段, 每行 '- 工具名: 描述'): {SEEDS}")
    print(f"[synth-tasks] 工具 {len(valid_tools)} 个 · 模型 {model} · 生成 {n} 个任务", flush=True)
    items, seen = [], set()
    for rnd in range(4):  # 不足 n 个就补轮, 最多 4 轮
        if len(items) >= n:
            break
        want = n - len(items)
        try:
            r = httpx.post(f"{base}/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "temperature": 0.8,
                                 "messages": [{"role": "user",
                                               "content": PROMPT.format(n=min(want, 6), tools=tools_txt,
                                                                        domain=domain, fewshot=few_shot())}]},
                           timeout=600)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"[synth-tasks] ✗ 第 {rnd + 1} 轮生成失败: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            continue
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        added = 0
        for t in arr if isinstance(arr, list) else []:
            task = _norm_task(t or {}, valid_tools)
            if task and task["instruction"] not in seen:
                seen.add(task["instruction"])
                items.append(task)
                added += 1
        print(f"[synth-tasks] 第 {rnd + 1} 轮 +{added}, 累计 {len(items)}/{n}", flush=True)
    return items[:n]


def write_pending(items: list[dict]):
    pending = read_jsonl(PENDING)
    next_pid = max((int(x.get("id", 0)) for x in pending), default=0)
    rows = pending + [{"id": next_pid + i + 1, **it} for i, it in enumerate(items)]
    tmp = PENDING.with_suffix(".jsonl.tmp")
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    tmp.replace(PENDING)


def selftest():
    write_pending([
        {"instruction": "(自测)检索'量子计算'的最新进展, 回答时引用一句资料。",
         "expect_tools": ["web_search"], "expect_answer_contains": ["量子"],
         "rationale": "自测: 单检索型"},
        {"instruction": "(自测)调查'可再生能源'现状, 把结论写入工作区文件 可再生能源报告.md。",
         "expect_tools": ["web_search", "write_file"], "expect_file": "可再生能源报告.md",
         "rationale": "自测: 检索+写文件组合型"},
    ])
    print(f"[synth-tasks] selftest 完成: 2 条已写入待审区 {PENDING}", flush=True)


def main():
    load_env_file()
    if SELFTEST:
        selftest()
        return
    n = max(1, min(20, int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5))
    items = llm_generate(n)
    if not items:
        sys.exit("[synth-tasks] 一个任务都没生成成功, 看上方日志")
    write_pending(items)
    print(f"[synth-tasks] 完成: {len(items)} 条待审任务已写入 {PENDING}, 去平台审核采纳", flush=True)


if __name__ == "__main__":
    main()
