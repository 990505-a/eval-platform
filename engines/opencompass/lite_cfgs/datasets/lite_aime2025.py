# -*- coding: utf-8 -*-
"""AIME 2025 (I/II 卷全量 30 题) · 手写配置(非截断生成).

与官方 aime2024_gen_17d799 同构, 差异仅两点:
1) 数据源: 本地 ./data/aime2025/aime2025.jsonl (HF opencompass/AIME2025 下载合并);
2) 判卷: AimeIntEvaluator 整数精确匹配(抽 \\boxed{} 与金标比对, 公开榜同口径)。
   官方 MATHVerifyEvaluator 在 Windows 上每题子进程加载 sympy 超过其 10s join
   超时, 30 题全部被误判超时算错(flash 实测 0.00), 故弃用; 评估器由 run_bench.py
   的 ensure_aime_patch 幂等内嵌进 opencompass, 此处直接 import 类对象 ——
   字符串引用在 openicl_eval 的判卷构建路径查不到注册表会得到 None(实测崩溃)。
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator.math_evaluator import AimeIntEvaluator
from opencompass.datasets import CustomDataset

aime2025_reader_cfg = dict(input_columns=['question'], output_column='answer')

aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN',
                     prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

aime2025_eval_cfg = dict(evaluator=dict(type=AimeIntEvaluator))

aime2025_datasets = [
    dict(
        abbr='aime2025',
        type=CustomDataset,
        path='./data/aime2025/aime2025.jsonl',
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        eval_cfg=aime2025_eval_cfg)
]
