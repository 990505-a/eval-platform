# -*- coding: utf-8 -*-
"""C-Eval 智能答案提取 —— 供 lite_ceval.py 动态配置引用.

opencompass 默认的 first_capital_postprocess 取回答里"第一个大写字母", 对英文作答或
先分析后作答的模型几乎全判错(实测 glm-5.3 从 89.2% 被误判成 1.9%); 这里改为:
思考链拼接文本(推理…</think>最终答案)优先只看最终段 —— 思考模型(DeepSeek 中文 CoT)
草稿里全是中间假设字母, 取"第一次出现"会抓到假设而不是结论;
显式答案标记(中文/英文)取全文中最后一次出现; 都没有则取最后一个 A-D 字母。
"""
import re

_PATTERNS = (r"答案是?\s*[:：]?\s*([A-D])", r"[Cc]orrect\s+answer\s+is\s*([A-D])",
             r"[Aa]nswer\s*[:：]\s*([A-D])", r"因此选\s*([A-D])", r"所以选\s*([A-D])",
             r"选项\s*([A-D])")


def smart_ceval_postprocess(text: str) -> str:
    if not text:
        return ""
    # 思考链文本: 最终答案在 </think> 之后, 直接取该段的字母
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1]
        for ch in reversed(tail):
            if ch in "ABCD":
                return ch
    # 显式答案标记: 取最后一次出现(前面的是草稿里的中间假设)
    best = None
    for pat in _PATTERNS:
        for m in re.finditer(pat, text):
            best = m.group(1).upper()
    if best:
        return best
    for ch in reversed(text):
        if ch in "ABCD":
            return ch
    return ""
