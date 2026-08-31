# -*- coding: utf-8 -*-
"""智能体轨迹评分 (DeepEval 双口径): 最新轨迹 → agent_scores.json + 归档

线B 评分引擎 (替代 ragas agent 指标, ragas 收缩回纯 RAG):
  裁判式 (DeepEval GEval, 任意 OpenAI 兼容端点当裁判):
    tool_correctness  实际工具调用 vs 期望工具集合 (多调/漏调都扣分)
    task_completion   任务目标最终是否达成 (对照期望要点判 0-1)
  确定性核验 (本脚本自算, 不花 API):
    tool_recall / answer_hit / file_hit / memory_hit (环境终态证据来自 run_tasks.py 采集)

产物契约 (与平台 app.py / 前端对齐):
    runs/agenteval/agent_scores.json          当前分数
    runs/agenteval/archive/<ts>.json          历史轮次(自动归档)

用法:
    python evaluate.py             # 正常评分(裁判走 OPENAI_* 端点)
    python evaluate.py --selftest  # 跳过裁判, 只算确定性核验(链路验证, 零 API)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "runs" / "agenteval"
SELFTEST = "--selftest" in sys.argv


# ---------- 裁判模型: 任意 OpenAI 兼容端点包装成 DeepEval 自定义模型 ----------
def build_judge():
    import os
    from deepeval.models import DeepEvalBaseLLM

    model = os.environ.get("AGENTEVAL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("裁判未配置: 缺 OPENAI_API_KEY (⚙️设置/模块设置-agent 或 .env)")

    class JudgeLLM(DeepEvalBaseLLM):
        def load_model(self):
            return self

        def get_model_name(self):
            return model

        def generate_text(self, prompt: str) -> str:
            r = httpx.post(f"{base}/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "temperature": 0,
                                 "messages": [{"role": "user", "content": prompt}]},
                           timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    print(f"[evaluate] 裁判: {model} @ {base}", flush=True)
    return JudgeLLM(), model


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _det(traj: dict) -> dict:
    """确定性核验: 对照期望标注 + 环境终态证据, 不花 API。"""
    exp, called = traj["expect"], traj.get("pred_tools") or []
    out = {"tool_recall": None, "answer_hit": None, "file_hit": None, "memory_hit": None}
    if exp["tools"]:
        hit = {t for t in exp["tools"] if t in called}
        out["tool_recall"] = round(len(hit) / len(set(exp["tools"])), 3)
    if exp["answer_contains"]:
        reply = traj.get("reply") or ""
        out["answer_hit"] = round(sum(1 for kw in exp["answer_contains"] if kw in reply)
                                  / len(exp["answer_contains"]), 3)
    if exp["file"]:
        files = traj.get("files") or []
        out["file_hit"] = any(exp["file"] in f for f in files)
    if exp["memory_contains"]:
        out["memory_hit"] = exp["memory_contains"] in (traj.get("memory") or "")
    return out


def _judge(traj: dict, judge) -> dict:
    """DeepEval 裁判两项: 工具正确性 + 任务完成度。"""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    exp = traj["expect"]
    called = traj.get("pred_tools") or []
    expect_notes = []
    if exp["answer_contains"]:
        expect_notes.append(f"回答需包含: {exp['answer_contains']}")
    if exp["file"]:
        expect_notes.append(f"需产出文件: {exp['file']}")
    if exp["memory_contains"]:
        expect_notes.append(f"需写入长期记忆: {exp['memory_contains']}")
    evidence = (f"最终回答: {traj.get('reply') or '(空)'}\n"
                f"工作区文件: {traj.get('files') or '无'}\n"
                f"长期记忆末尾: {(traj.get('memory') or '无')[-200:]}")
    out = {}
    cases = {
        "tool_correctness": (
            f"任务的期望工具集合: {exp['tools'] or '(无标注)'}。"
            "判断实际调用的工具是否恰当: 该用的用了没有、有没有调用与任务无关的多余工具。"
            "完全正确=1, 漏调或多调明显不当≈0.5, 完全不对=0。只评工具名与必要性, 不评参数。",
            f"实际调用: {called or '无'}"),
        "task_completion": (
            f"任务指令: {traj['instruction']}\n期望要点: {'; '.join(expect_notes) or '按指令判断'}。"
            "综合最终回答与工作区证据, 判断任务目标是否达成。完全达成=1, 部分达成≈0.5, 未达成=0。",
            evidence),
    }
    for key, (criteria, actual) in cases.items():
        m = GEval(name=key, criteria=criteria, model=judge, threshold=0.5,
                  evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT])
        m.measure(LLMTestCase(input=criteria, actual_output=actual))
        out[key] = round(float(m.score), 3) if m.score is not None else None
        out[key + "_reason"] = (m.reason or "")[:200]
        print(f"[evaluate]   任务{traj['id']} {key}={out[key]}", flush=True)
    return out


def main():
    trajs_files = sorted(RUNS.glob("trajectories-*.jsonl"), reverse=True)
    if not trajs_files:
        raise SystemExit("没有轨迹: 先跑 run_tasks.py (平台「生成轨迹」)")
    src = trajs_files[0]
    trajs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[evaluate] 评分 {len(trajs)} 题 · 来源 {src.name} · "
          f"{'selftest(裁判跳过)' if SELFTEST else 'DeepEval 裁判'}", flush=True)

    judge = None
    if not SELFTEST:
        judge, judge_name = build_judge()
    else:
        judge_name = "selftest"

    per_task = []
    for t in trajs:
        row = {"task_id": t["id"], "instruction": t["instruction"],
               "pred_tools": t.get("pred_tools"), "expect_tools": t["expect"]["tools"],
               "latency_s": t.get("latency_s"), **_det(t)}
        if SELFTEST:
            row.update({"tool_correctness": None, "task_completion": None})
        else:
            try:
                row.update(_judge(t, judge))
            except Exception as e:  # noqa: BLE001  # 裁判单题失败不中断
                print(f"[evaluate]   任务{t['id']} 裁判失败: {e}", flush=True)
                row.update({"tool_correctness": None, "task_completion": None})
        per_task.append(row)

    summary = {k: _mean([r.get(k) for r in per_task])
               for k in ("tool_correctness", "task_completion",
                         "tool_recall", "answer_hit", "file_hit", "memory_hit")}
    scores = {
        "engine": "deepeval", "judge": judge_name, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(per_task), "source": src.name, "summary": summary,
        "mean_latency_s": _mean([r.get("latency_s") for r in per_task]),
        "per_task": per_task,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "agent_scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    arch = RUNS / "archive" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    arch.parent.mkdir(exist_ok=True)
    arch.write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[evaluate] 完成: summary={summary}", flush=True)
    print(f"[evaluate] 归档: {arch.name}", flush=True)


if __name__ == "__main__":
    main()
