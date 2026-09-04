# -*- coding: utf-8 -*-
"""自定义智能体轨迹评分 (DeepEval 双口径): 最新轨迹 → agent_scores.json + 归档

评分引擎 (Langfuse 记轨迹, DeepEval 打分 —— 官方推荐的分工):
  裁判式 (DeepEval GEval, 任意 OpenAI 兼容端点当裁判, 见 judge.py):
    tool_correctness  实际工具调用(含参数与顺序) vs 期望工具集合 (多调/漏调都扣分)
    task_completion   任务目标最终是否达成 (对照期望要点判 0-1)
  确定性核验 (本脚本自算, 不花 API):
    tool_recall / answer_hit / file_hit / content_hit (环境终态证据来自 run_tasks.py 采集)

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

from judge import build_judge

HERE = Path(__file__).parent
RUNS = HERE.parent.parent / "runs" / "agenteval"
SELFTEST = "--selftest" in sys.argv


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _det(traj: dict) -> dict:
    """确定性核验: 对照期望标注 + 环境终态证据, 不花 API。"""
    exp, called = traj["expect"], traj.get("pred_tools") or []
    out = {"tool_recall": None, "answer_hit": None, "file_hit": None, "content_hit": None}
    if exp.get("tools"):
        hit = {t for t in exp["tools"] if t in called}
        out["tool_recall"] = round(len(hit) / len(set(exp["tools"])), 3)
    if exp.get("answer_contains"):
        reply = traj.get("reply") or ""
        out["answer_hit"] = round(sum(1 for kw in exp["answer_contains"] if kw in reply)
                                  / len(exp["answer_contains"]), 3)
    if exp.get("file"):
        out["file_hit"] = any(exp["file"] in f for f in traj.get("files") or [])
    # 内容核验: expect_file_contains{file,text}; 旧轨迹的 memory 字段/memory_contains 兼容
    fc = exp.get("file_contains")
    if fc:
        ev = ((traj.get("file_evidence") or {}).get(fc["file"])
              if isinstance(fc, dict) else None)
        out["content_hit"] = bool(fc.get("text")) and str(fc["text"]) in (ev or "")
    elif exp.get("memory_contains"):
        ev = ((traj.get("file_evidence") or {}).get("memory/long_term_memory.md")
              or traj.get("memory") or "")
        out["content_hit"] = str(exp["memory_contains"]) in ev
    return out


def _calls_text(traj: dict) -> str:
    """带参数、按序的工具调用序列(喂给裁判的完整过程信息)。"""
    calls = traj.get("tool_calls") or []
    if not calls:
        return "无"
    return " → ".join(f"{c.get('name', '?')}({str(c.get('args', ''))[:80]})" for c in calls)


def _judge(traj: dict, judge) -> dict:
    """DeepEval 裁判两项: 工具正确性 + 任务完成度。"""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    exp = traj["expect"]
    det = traj.get("_det") or {}
    expect_notes = []
    if exp.get("answer_contains"):
        expect_notes.append(f"回答需包含: {exp['answer_contains']}")
    if exp.get("file"):
        expect_notes.append(f"需产出文件: {exp['file']}")
    if exp.get("file_contains"):
        expect_notes.append(f"文件 {exp['file_contains'].get('file')} 需包含: {exp['file_contains'].get('text')}")
    if exp.get("memory_contains"):
        expect_notes.append(f"需写入长期记忆: {exp['memory_contains']}")
    evidence = (f"最终回答: {traj.get('reply') or '(空)'}\n"
                f"工作区文件: {traj.get('files') or '无'}\n"
                f"文件内容证据: {traj.get('file_evidence') or '无'}\n"
                f"执行异常: {traj.get('error') or '无'}")
    out = {}
    cases = {
        "tool_correctness": (
            f"任务的期望工具集合: {exp.get('tools') or '(无标注)'}。"
            "下面是智能体按顺序实际发起的工具调用(含参数摘要), 判断工具选择是否恰当: "
            "该用的用了没有、顺序合理吗、有没有与任务无关的多余调用。"
            "完全正确=1, 漏调或明显多调不当≈0.5, 完全不对=0。",
            f"实际调用序列: {_calls_text(traj)}"),
        "task_completion": (
            f"任务指令: {traj['instruction']}\n期望要点: {'; '.join(expect_notes) or '按指令判断'}。\n"
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
        raise SystemExit("没有轨迹: 先跑 run_tasks.py (平台「一键评测」)")
    src = trajs_files[0]
    trajs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[evaluate] 评分 {len(trajs)} 题 · 来源 {src.name} · "
          f"{'selftest(裁判跳过)' if SELFTEST else 'DeepEval 裁判'}", flush=True)

    judge = None
    judge_name = "selftest"
    if not SELFTEST:
        try:
            judge, judge_name = build_judge()
        except SystemExit as e:  # 无裁判 key → 降级为仅确定性核验, 不中断
            print(f"[evaluate] {e}\n[evaluate] → 本轮跳过裁判, 仅算确定性核验"
                  "(tool_recall/answer_hit/file_hit/content_hit), 裁判项记 None", flush=True)
            judge_name = "no-key(仅确定性核验)"

    per_task = []
    for t in trajs:
        det = _det(t)
        row = {"task_id": t["id"], "instruction": t["instruction"],
               "pred_tools": t.get("pred_tools"), "expect_tools": t["expect"].get("tools"),
               "latency_s": t.get("latency_s"), "error": t.get("error"), **det}
        if SELFTEST or judge is None:
            row.update({"tool_correctness": None, "task_completion": None})
        else:
            try:
                t["_det"] = det
                row.update(_judge(t, judge))
            except Exception as e:  # noqa: BLE001  # 裁判单题失败不中断
                print(f"[evaluate]   任务{t['id']} 裁判失败: {e}", flush=True)
                row.update({"tool_correctness": None, "task_completion": None})
        per_task.append(row)

    summary = {k: _mean([r.get(k) for r in per_task])
               for k in ("tool_correctness", "task_completion",
                         "tool_recall", "answer_hit", "file_hit", "content_hit")}
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
