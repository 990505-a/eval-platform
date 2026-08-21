# -*- coding: utf-8 -*-
"""OpenCompass 基准配置: gpt-4o-mini x C-Eval (52 学科, val 每科 5 题, API 模型).

注: mmengine 配置加载用懒导入, 无法在配置内切片数据集 —— 全量 260 题反而给出完整 C-Eval 分.
"""
from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets  # noqa: F401
from opencompass.models import OpenAI

api_meta_template = dict(round=[
    dict(role="HUMAN", api_role="HUMAN"),
    dict(role="BOT", api_role="BOT", generate=True),
])

models = [
    dict(
        abbr="gpt-4o-mini",
        type=OpenAI,
        path="gpt-4o-mini",
        key="ENV",  # 读 OPENAI_API_KEY 环境变量
        meta_template=api_meta_template,
        # 显式传完整端点: 绕过 opencompass 默认 os.path.join 在 Windows 上的反斜杠(%5C) 404。
        # 本文件是示例配置; 自定义端点请走 run_bench.py(动态配置, 端点由环境变量计算内联)
        openai_api_base='https://api.openai.com/v1/chat/completions',
        query_per_second=4,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=8,
    ),
]

datasets = ceval_datasets

summarizer = dict(roles=["assistant"], metric=["accuracy"])
