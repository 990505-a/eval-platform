# -*- coding: utf-8 -*-
"""自有题库合成: 从种子语料(data/own_seeds/)随机抽材料块 → LLM 依据材料出选择题 → 待审区.

用法 (平台经 sys.executable 拉起, 只依赖 httpx):
    python synth_bench.py [N]        # 默认 5, clamp 到 [1, 20]
    python synth_bench.py --selftest # 零 API 自测: 写 2 条硬编码题到待审区

约定 (与 ragas 的 synth_questions.py 同构):
- 待审区 data/own_synth.jsonl, 追加式 + id 续号 + 原子写(.tmp → replace)
- 每行: {id, question, A, B, C, D, answer, basis, source}   # basis = 材料原句锚, 审核时核对
- 采纳后由平台写入正式题库 data/own/own.jsonl (bench key: own)
- 进度一律 print("[synth-bench] ...", flush=True), 日志落 runs/llm_synth.log
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
SEEDS = HERE / "data" / "own_seeds"
PENDING = HERE / "data" / "own_synth.jsonl"

SELFTEST = "--selftest" in sys.argv


def load_env_file():
    """平台注入的环境变量优先, 缺的从平台根 .env setdefault 补齐."""
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


def pick_seed_blocks(n: int, selected: list[str] | None = None) -> list[tuple[str, str]]:
    """从指定或全部种子文件各抽一块 800-4000 字材料."""
    allowed = set(selected or [])
    files = [f for f in SEEDS.glob("*")
             if f.is_file() and f.suffix.lower() in (".txt", ".md") and f.stat().st_size > 200
             and (not allowed or f.name in allowed)]
    if not files:
        return []
    blocks = []
    for f in files:
        paras = [p.strip() for p in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")) if len(p.strip()) > 60]
        if not paras:
            continue
        # 顺序聚段到目标长度, 随机起点
        start = random.randrange(len(paras))
        buf, picked = [], []
        for i in range(start, start + len(paras)):
            picked.append(paras[i % len(paras)])
            if sum(map(len, picked)) >= random.randint(800, 4000):
                break
        text = "\n".join(picked)
        if len(text) > 4000:
            text = text[:4000]
        blocks.append((f.name, text))
    random.shuffle(blocks)
    return blocks[:max(n, 3)]


PROMPT = """下面是一段业务领域的材料。请依据这段材料出 {n} 道四选一的单项选择题, 用于评测大模型对该领域的掌握程度。

要求:
1. 题目必须仅凭这段材料可答, 不依赖外部知识; 考察事实、关系、细节, 不要出观点题
2. 干扰项必须合理(同类别、易混淆), 不能一眼假
3. answer 只能是 A/B/C/D 单字母, 且正确选项内容必须能在材料中找到依据
4. basis 字段摘录材料中支持答案的原句(15-60 字), 供人工审核核对
5. 难度分层: 约 1/3 直陈细节题, 1/3 需要理解转换, 1/3 需要跨段落综合
6. 只输出 JSON 数组, 不要任何解释: [{{"question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "answer": "A", "basis": "..."}}]

材料:
{chunk}"""


def llm_generate(n: int, selected: list[str] | None = None) -> list[dict]:
    key = os.environ.get("OPENAI_API_KEY", "")
    base = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not key:
        sys.exit("[synth-bench] 缺 OPENAI_API_KEY (平台 .env 或 ⚙️ 设置/llm 模块)")
    blocks = pick_seed_blocks(n, selected)
    if not blocks:
        sys.exit(f"[synth-bench] 没有可用的指定素材: {SEEDS}")
    print(f"[synth-bench] 从 {len(blocks)} 个种子文件抽材料, 模型 {model}, 出 {n} 题", flush=True)
    items, seen = [], set()
    for fname, chunk in blocks:
        if len(items) >= n:
            break
        try:
            r = httpx.post(f"{base}/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "temperature": 0.7,
                                 "messages": [{"role": "user",
                                               "content": PROMPT.format(n=min(2, n - len(items) + 1), chunk=chunk)}]},
                           timeout=600)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"[synth-bench] ✗ {fname} 生成失败: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            print(f"[synth-bench] ✗ {fname} 返回非 JSON, 跳过", flush=True)
            continue
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for q in arr if isinstance(arr, list) else []:
            q = {k: str(v).strip() for k, v in (q or {}).items()
                 if k in ("question", "A", "B", "C", "D", "answer", "basis")}
            ans = re.sub(r"[^A-D]", "", q.get("answer", ""))[:1].upper()
            if not q.get("question") or len(ans) != 1 or any(not q.get(c) for c in "ABCD"):
                continue
            if q["question"] in seen:
                continue
            seen.add(q["question"])
            q["answer"] = ans
            q["source"] = fname
            items.append(q)
        print(f"[synth-bench] {fname}: 累计 {len(items)} 题", flush=True)
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
        {"question": "(自测)《水浒传》的作者是谁?", "A": "罗贯中", "B": "施耐庵", "C": "吴承恩", "D": "曹雪芹",
         "answer": "B", "basis": "《水浒传》元末明初施耐庵著", "source": "selftest"},
        {"question": "(自测)莎士比亚四大悲剧不包括?", "A": "《哈姆雷特》", "B": "《麦克白》", "C": "《李尔王》", "D": "《威尼斯商人》",
         "answer": "D", "basis": "四大悲剧为《哈姆雷特》《奥赛罗》《李尔王》《麦克白》", "source": "selftest"},
    ])
    print(f"[synth-bench] selftest 完成: 2 条已写入待审区 {PENDING}", flush=True)


def main():
    load_env_file()
    if SELFTEST:
        selftest()
        return
    args = [a for a in sys.argv[1:] if a != "--files"]
    n = max(1, min(20, int(args[0]) if args and args[0].isdigit() else 5))
    selected = sys.argv[sys.argv.index("--files") + 1:] if "--files" in sys.argv else None
    items = llm_generate(n, selected)
    if not items:
        sys.exit("[synth-bench] 一题都没生成成功, 看上方日志")
    write_pending(items)
    print(f"[synth-bench] 完成: {len(items)} 条待审题已写入 {PENDING}, 去平台审核采纳", flush=True)


if __name__ == "__main__":
    main()
