# -*- coding: utf-8 -*-
"""RAGFlow 补丁: DeepSeek 系模型默认关闭思考(对齐官方 litellm-DeepSeek 家族策略)。
v0.27 的 OpenAI-API-Compatible 通道(base 后端)不传任何思考参数, DeepSeek v4 全系默认带思考;
本补丁在 _apply_model_family_policies 里为 deepseek* 模型注入 thinking.disabled。
用法: docker exec eval-platform-ragflow-ragflow-1 python3 /patch.py  (容器重建后需重打+重启)
注意: 改的是运行容器外文件, 需重启 ragflow 容器生效。"""
import re, sys
P = "/ragflow/rag/llm/chat_model.py"
src = open(P).read()
if "deepseek" in src and "_merge_extra_body(sanitized_kwargs, {\"thinking\"" in src:
    print("已打过补丁, 跳过"); sys.exit(0)
ANCHOR = "\n    if backend == \"base\":\n        return sanitized_gen_conf, sanitized_kwargs"
PATCH = ("\n    elif \"deepseek\" in model_name_lower:\n"
         "        _pop_thinking_controls()\n"
         "        _merge_extra_body(sanitized_kwargs, {\"thinking\": {\"type\": thinking_type or \"disabled\"}})\n"
         + ANCHOR)
assert src.count(ANCHOR) == 1, f"锚点异常: {src.count(ANCHOR)} 处"
open(P, "w").write(src.replace(ANCHOR, PATCH))
print("补丁完成, 重启 ragflow 容器后生效")
