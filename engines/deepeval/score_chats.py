# -*- coding: utf-8 -*-
"""对话轨迹抽样评分: Langfuse 拉轨迹 → DeepEval 打分 → 分数写回 Langfuse

官方 cookbook 的三步模式落地 ("Langfuse captures, DeepEval scores"):
    ① GET  {host}/api/public/traces?limit=N&orderBy=timestamp.desc   拉最近轨迹
    ② GEval(任务完成度/帮助性) 逐条打分 (裁判走共享 judge.py)
    ③ POST {host}/api/public/scores                                  分数+理由挂回原轨迹

产物: runs/agenteval/chatscores.json (平台「轨迹回放」页签展示)
环境变量: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST (平台注入 langfuse.env)
          AGENTEVAL_CHAT_N 抽样条数(默认 5) · OPENAI_* 裁判端点
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

from judge import build_judge as _shared_judge

HERE = Path(__file__).parent
RUNS = HERE.parent.parent / "runs" / "agenteval"

PK = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
SK = os.environ.get("LANGFUSE_SECRET_KEY", "")
HOST = (os.environ.get("LANGFUSE_HOST") or "").rstrip("/")
N = int(os.environ.get("AGENTEVAL_CHAT_N") or "5")


def _text(v) -> str:
    """Langfuse 轨迹的 input/output 可能是 str/list/dict, 统一成可读文本。"""
    if v is None:
        return "(空)"
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def fetch_traces() -> list[dict]:
    auth = (PK, SK)
    for url in (f"{HOST}/api/public/traces?limit={N}&orderBy=timestamp.desc",
                f"{HOST}/api/public/traces?limit={N}"):
        r = httpx.get(url, auth=auth, timeout=30)
        if r.status_code == 200:
            return r.json().get("data") or []
    raise SystemExit(f"拉取轨迹失败 HTTP {r.status_code}: {r.text[:200]}")


def build_judge():
    return _shared_judge()


def main():
    if not (PK and SK and HOST):
        raise SystemExit("未配置 Langfuse: 需 LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST (langfuse/langfuse.env)")
    traces = fetch_traces()
    if not traces:
        _save({"ts": _now(), "n": 0, "mean": None, "items": [],
               "note": f"Langfuse ({HOST}) 最近无轨迹"})
        print("[score_chats] Langfuse 无轨迹, 先去对话页聊几轮", flush=True)
        return
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    judge, judge_name = build_judge()
    print(f"[score_chats] 抽样 {len(traces)} 条 · 裁判 {judge_name} · {HOST}", flush=True)
    metric = GEval(
        name="answer_quality", model=judge, threshold=0.5,
        criteria="你是智能体回复质量评审。给定用户输入与智能体最终回复, 综合判断: 任务完成度、"
                 "对用户请求的帮助性、表述清晰度。完全达标=1, 明显不足=0, 介于之间线性给分。"
                 "只依据给出的文本判断, 不要臆测未展示的过程。",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT])

    items, ok = [], 0
    for t in traces:
        try:
            metric.measure(LLMTestCase(input=_text(t.get("input")),
                                       actual_output=_text(t.get("output"))))
            score = round(float(metric.score), 3) if metric.score is not None else None
            reason = (metric.reason or "")[:300]
            if score is None:
                continue
            wr = httpx.post(f"{HOST}/api/public/scores", auth=(PK, SK), timeout=30,
                            json={"traceId": t["id"], "name": "deepeval-answer-quality",
                                  "value": score, "comment": reason})
            ok += 1 if wr.status_code in (200, 201) else 0
            items.append({"trace_id": t["id"], "name": t.get("name") or "",
                          "ts": (t.get("timestamp") or "")[:19], "score": score,
                          "reason": reason, "writeback": wr.status_code})
            print(f"[score_chats] {t['id'][:12]}… score={score} 写回={wr.status_code}", flush=True)
        except Exception as e:  # noqa: BLE001  # 单条失败不中断
            print(f"[score_chats] 失败 {t.get('id')}: {e}", flush=True)
    mean = round(sum(i["score"] for i in items) / len(items), 3) if items else None
    _save({"ts": _now(), "judge": judge_name, "n": len(items),
           "mean": mean, "writeback_ok": ok, "host": HOST, "items": items})
    print(f"[score_chats] 完成: {len(items)} 条 · 均分 {mean} · 写回成功 {ok}", flush=True)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _save(d: dict):
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "chatscores.json").write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
