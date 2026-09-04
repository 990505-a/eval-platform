# -*- coding: utf-8 -*-
"""共享裁判: 任意 OpenAI 兼容端点包装成 DeepEval 自定义模型.

evaluate.py (任务集轨迹评分) 与 score_chats.py (对话流量抽样评分) 共用,
改裁判配置/超时只改这一处.

环境变量(平台注入, 回退平台 .env):
    AGENTEVAL_JUDGE_MODEL > OPENAI_MODEL > 默认 gpt-4o-mini
    OPENAI_BASE_URL / OPENAI_API_KEY
"""
import os

import httpx


def judge_target() -> tuple[str, str, str]:
    """(model, base_url, api_key) —— 裁判端点解析."""
    model = os.environ.get("AGENTEVAL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    base = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    return model, base, key


def build_judge():
    """DeepEval 自定义裁判模型; 未配置 key 时 SystemExit 带可读提示."""
    from deepeval.models import DeepEvalBaseLLM

    model, base, key = judge_target()
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

    print(f"[judge] 裁判: {model} @ {base}", flush=True)
    return JudgeLLM(), model
