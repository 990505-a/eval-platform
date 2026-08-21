# 动态生成: CMMLU 限题版 (path 指向 data/cmmlu_lite, 每子集前 5 题). 勿手改
from mmengine.config import read_base

with read_base():
    from .cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403