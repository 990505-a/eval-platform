# -*- coding: utf-8 -*-
r"""AI 合成题集(带黄金段落): 先从 RAGFlow 索引随机抽真实原文块 → LLM 从块出题 → 块作为黄金段落。

产出写入待审区 data/questions_synth.jsonl, 前端「题集管理」采纳/丢弃, 不直接进题库。
每条: {id, book, user_input, reference, reference_contexts: [块原文], source: 文档名}
锚保证真实(直接来自索引块), 合成题与手写题同等可信。

与平台契约: /api/rag/synth 以 argv=N 拉起本脚本, 进程检测关键字 synth_questions。
用法: synth_questions.py [N]   (默认 5, 上限 20; 仅 ragflow 后端支持配锚)
"""
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PENDING = DATA / "questions_synth.jsonl"

sys.path.insert(0, str(HERE))
from retrieve import RagFlow, load_env_file  # noqa: E402  复用登录/数据集解析与 .env 读取

PROMPT = """下面是名著中的一段原文。请依据这段原文出 {n} 道事实型问答题, 用于 RAG 检索评测。
要求:
1. 问题必须能**仅凭这段原文**回答(不得需要块外知识), 考这块里的具体人物/事件/因果/细节
2. reference 为要点式参考答案(2-4句), 只含原文中的事实, 关键人名/数字/物件保留原文措辞
3. 只输出 JSON 数组: [{{"user_input": "...", "reference": "..."}}, ...]

【原文】
{chunk}"""


def main():
    load_env_file()
    n = max(1, min(20, int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5))

    rf = RagFlow()
    rf._login()
    rf.ensure_dataset()
    # 列数据集内的文档 → 每本随机抽一块(长度合格), 分散在多本书
    import httpx
    docs_r = rf._get(f"/api/v1/datasets/{rf.dataset_id}/documents", page=1, page_size=50)
    docs = docs_r.get("data") if isinstance(docs_r.get("data"), list) else (docs_r.get("data") or {}).get("docs") or []
    chunks = []
    for doc in docs:
        total_pages = max(1, (doc.get("chunk_count") or 1) // 50 + 1)
        r = rf._get(f"/api/v1/datasets/{rf.dataset_id}/documents/{doc['id']}/chunks",
                    page=random.randint(1, total_pages), page_size=50)
        cs = (r.get("data") or {}).get("chunks") if isinstance(r.get("data"), dict) else r.get("data") or []
        good = [c for c in cs if len(c.get("content") or "") > 200]
        if good:
            chunks.append((doc["name"], random.choice(good)["content"].strip()))
    if not chunks:
        print("[synth] 未能从索引抽到任何块, 中止", flush=True)
        return
    random.shuffle(chunks)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                    base_url=(os.environ.get("OPENAI_BASE_URL")
                              or "https://api.openai.com/v1").rstrip("/"), timeout=600)
    model = os.environ.get("OPENAI_MODEL") or "glm-4.7"

    pending = []
    if PENDING.exists():
        for line in PENDING.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    pending.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    next_pid = max((int(p.get("id", 0)) for p in pending), default=0) + 1

    added, seen = 0, set()
    for doc_name, chunk in chunks:
        if added >= n:
            break
        book = re.sub(r"\.txt$", "", doc_name)
        if book == "shakespeare_complete":
            book = "莎士比亚"
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.7,
                messages=[{"role": "user", "content": PROMPT.format(n=2, chunk=chunk[:3000])}])
            m = re.search(r"\[.*\]", (r.choices[0].message.content or "").strip(), re.DOTALL)
            items = json.loads(m.group(0)) if m else []
        except Exception as e:  # noqa: BLE001
            print(f"[synth] {doc_name} 出题失败: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        for it in items[:2]:
            q, ref = (it.get("user_input") or "").strip(), (it.get("reference") or "").strip()
            if not q or not ref or q in seen:
                continue
            seen.add(q)
            pending.append({"id": next_pid, "book": book, "user_input": q, "reference": ref,
                            "reference_contexts": [chunk], "source": doc_name})
            next_pid += 1
            added += 1
            if added >= n:
                break
        print(f"[synth] {doc_name} 出题完成, 累计 {added}/{n}", flush=True)

    PENDING.parent.mkdir(parents=True, exist_ok=True)
    tmp = PENDING.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pending) + "\n", encoding="utf-8")
    tmp.replace(PENDING)
    print(f"[synth] 新增 {added} 题到待审区(全部带黄金段落+来源), 前端「题集管理」采纳或丢弃", flush=True)


if __name__ == "__main__":
    main()
