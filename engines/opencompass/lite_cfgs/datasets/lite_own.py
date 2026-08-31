# -*- coding: utf-8 -*-
"""自有题库 (own) · 手写配置(非截断生成), kind="custom" 通道。

数据: ./data/own/own.jsonl, 每行 {question, A, B, C, D, answer(ABCD 单字母)},
     在平台 🎓 LLM 基准页「🧩 自有题库」在线增删或 AI 合成采纳后生成。
判分: AccEvaluator 字符串比对 + smart_ceval_postprocess 智能答案提取 —— 与 C-Eval 同款,
     该提取器由 run_bench.py 的 ensure_postprocessor_patch 幂等内嵌进 opencompass
     (取 </think> 后尾段 / 显式答案标记取最后一次 / 全文最后一个 A-D), 故此处用字符串全路径引用。
模板: 照抄 lite_ceval 的 MCQ 提示词, 去掉 few-shot(自有题库无 dev 集可取示例)改 ZeroRetriever。
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CustomDataset

own_reader_cfg = dict(input_columns=['question', 'A', 'B', 'C', 'D'], output_column='answer')

own_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN',
                     prompt='以下是关于自有业务领域的单项选择题，请选出其中的正确答案。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请只输出正确选项的字母(A/B/C/D)，不要输出任何解释。\n答案: '),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

own_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type='opencompass.utils.text_postprocessors.smart_ceval_postprocess'))

own_datasets = [
    dict(
        abbr='own',
        type=CustomDataset,
        path='./data/own/own.jsonl',
        reader_cfg=own_reader_cfg,
        infer_cfg=own_infer_cfg,
        eval_cfg=own_eval_cfg)
]
