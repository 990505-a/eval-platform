# -*- coding: utf-8 -*-
"""C-Eval 答卷重判(智能提取): opencompass 默认取"第一个大写字母"判卷, 对用英文作答的
模型几乎全判错; 本脚本用显式答案标记优先、否则取最后一个 A-D 的方式重判已有 predictions,
不产生任何 API 费用。产物: 更新 latest.json 成绩矩阵 + 纠正 history.jsonl/archive 记录。

用法: .venv-opencompass\\Scripts\\python.exe rescore_ceval.py [模型名]
"""
import csv
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

PLATFORM = Path(__file__).resolve().parent.parent.parent
WORK = PLATFORM / "runs" / "llm" / "work"
VAL_DIR = Path(__file__).resolve().parent / "data" / "ceval" / "formal_ceval" / "val"
LATEST = PLATFORM / "runs" / "llm" / "latest.json"
HISTORY = PLATFORM / "runs" / "llm" / "history.jsonl"
ARCHIVE = PLATFORM / "runs" / "llm" / "archive"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "glm-5.3"

# 提取逻辑与判卷配置同源: 思考链文本取 </think> 后最终段, 显式标记取最后一次出现
from smart_eval import smart_ceval_postprocess as smart_extract


def main() -> None:
    exp_dirs = sorted([d for d in WORK.glob("*/predictions") if (d / MODEL).is_dir()],
                      key=lambda d: d.stat().st_mtime)
    if not exp_dirs:
        sys.exit(f"没有 {MODEL} 的 predictions 可重判")
    pred_dir = exp_dirs[-1].parent
    print(f"[rescore] 重判 {MODEL} @ {pred_dir.name}")

    per_subject, n_total, n_agree = {}, 0, 0
    for pf in sorted((pred_dir / "predictions" / MODEL).glob("ceval-*.json")):
        subj = pf.stem[len("ceval-"):]
        csvp = VAL_DIR / f"{subj}_val.csv"
        golds = []
        with open(csvp, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                golds.append(row["answer"].strip().upper())
        pred = json.loads(pf.read_text(encoding="utf-8"))
        agree = 0
        for i, g in enumerate(golds):
            ex = smart_extract((pred.get(str(i)) or {}).get("prediction") or "")
            n_total += 1
            if ex == g:
                agree += 1
        n_agree += agree
        per_subject[subj] = round(agree / len(golds), 4)
    mean = round(n_agree / n_total, 4)
    print(f"[rescore] 命中 {n_agree}/{n_total} = {mean * 100:.1f}%")

    # 1) 更新成绩矩阵
    matrix = json.loads(LATEST.read_text(encoding="utf-8")) if LATEST.exists() else {}
    for subj, acc in per_subject.items():
        matrix.setdefault(MODEL, {})[f"ceval-{subj}"] = {"accuracy": acc}
    LATEST.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rescore] 已更新 {LATEST}")

    # 2) 纠正 history.jsonl 里该模型最近一次未重判过的 C-Eval 记录(改 mean + 标注重判)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if HISTORY.exists():
        lines = HISTORY.read_text(encoding="utf-8").splitlines()
        target = None
        for i, line in enumerate(lines):
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            if h.get("model") == MODEL and h.get("bench") == "C-Eval" and not h.get("rescore"):
                target = i  # 最后一次匹配
        out = []
        for i, line in enumerate(lines):
            if i == target:
                h = json.loads(line)
                h["mean"] = mean
                h["rescore"] = True
                h["rescore_ts"] = ts
                out.append(json.dumps(h, ensure_ascii=False))
            else:
                out.append(line)
        if target is None:
            out.append(json.dumps({"ts": ts, "model": MODEL, "bench": "C-Eval",
                                   "n": n_total, "mean": mean, "rescore": True},
                                  ensure_ascii=False))
        HISTORY.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[rescore] 已纠正 {HISTORY}")

    # 3) 纠正本轮归档(成绩明细换重判口径)
    if ARCHIVE.exists():
        for ap in sorted(ARCHIVE.glob("*.json"), reverse=True):
            try:
                a = json.loads(ap.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if a.get("model") == MODEL and a.get("bench") == "C-Eval" and not a.get("rescore"):
                a["mean"] = mean
                a["scores"] = {f"ceval-{k}": {"accuracy": v} for k, v in per_subject.items()}
                a["rescore"] = True
                a["rescore_ts"] = ts
                ap.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"[rescore] 已纠正归档 {ap.name}")
                break


if __name__ == "__main__":
    main()
