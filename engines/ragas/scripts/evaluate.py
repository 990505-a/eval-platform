# -*- coding: utf-8 -*-
r"""ragas 打分胶水: data/eval_<mode>.jsonl → results/scores_cache.json (断点缓存) + scores_<mode>.json + eval_report.md

与平台契约:
  scores_cache.json 的 key = f"{mode}@{eval文件md5前8}:{qid}:{metric}" — 平台 parse_cache 只认
  当前答卷指纹的分数, 重新检索后旧分自动失效; 逐条原子写盘, 中断续跑不重复扣费。

指标 (17 项主流程, 与平台 EVAL_N_METRICS / METRIC_LABELS 对齐):
  裁判 9: faithfulness answer_relevancy context_precision context_recall context_entity_recall
          factual_correctness answer_correctness semantic_similarity response_groundedness
  本地 5: rougeL bleu chrf string_similarity exact_match (零 API 成本)
  裁判 3: rubric_accuracy / rubric_completeness / rubric_grounding (1-5 分制, 自定义提示词)
  可选:   noise_sensitivity (--with-noise, 专项补)

环境: OPENAI_API_KEY/OPENAI_BASE_URL (裁判, 平台注入) / EVAL_JUDGE_MODEL (默认 OPENAI_MODEL|glm-4.7)
      EVAL_EMB_URL (默认本机 MLX http://127.0.0.1:7997/v1, 零费用) / EVAL_EMB_MODEL
用法: evaluate.py [mode ...] [--with-noise] [--limit N]
"""
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
CACHE = RESULTS / "scores_cache.json"

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall",
                 "context_entity_recall", "factual_correctness", "answer_correctness",
                 "semantic_similarity", "response_groundedness")
NEEDS_REFERENCE = {"context_precision", "context_recall", "context_entity_recall",
                   "factual_correctness", "answer_correctness", "semantic_similarity",
                   "rougeL", "bleu", "chrf", "exact_match", "string_similarity",
                   "rubric_accuracy", "rubric_completeness"}
RUBRICS = {"rubric_accuracy": "准确性: 回答内容与参考答案的事实一致程度, 有无编造或错误",
           "rubric_completeness": "完整性: 参考答案的要点被覆盖了多少, 有无重要遗漏",
           "rubric_grounding": "有据性: 回答是否基于所给检索资料并标注来源编号, 而非凭空发挥"}

RUBRIC_PROMPT = """你是测评裁判。请给回答打分(1-5 整数)并说明理由。
评分维度: {dim}
【问题】{q}
【参考答案】{ref}
【回答】{resp}
只输出 JSON: {{"score": <1-5>, "reason": "<一句话理由>"}}"""


def load_env_file():
    """独立手跑时读平台 .env (平台拉起时已注入环境变量, 这里不覆盖)。"""
    envf = ROOT.parent.parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def discover_modes() -> list[str]:
    if not DATA.exists():
        return []
    return sorted({p.stem.removeprefix("eval_") for p in DATA.glob("eval_*.jsonl")} - {"questions"})


def read_rows(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def content_fp(mode: str) -> str | None:
    p = DATA / f"eval_{mode}.jsonl"
    return hashlib.md5(p.read_bytes()).hexdigest()[:8] if p.exists() else None


def load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict):
    RESULTS.mkdir(exist_ok=True)
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE)


def judge_extra_body() -> dict:
    """裁判默认关思考(判分快且省token)。智谱/DeepSeek 域名用 thinking.disabled(实测两者都接受);
    其他网关不认识该参数会 400, 不发送。EVAL_JUDGE_THINKING=on 恢复思考。"""
    if os.environ.get("EVAL_JUDGE_THINKING", "").lower() in ("on", "1", "true"):
        return {}
    base = (os.environ.get("OPENAI_BASE_URL") or "").lower()
    if "bigmodel" in base or "zhipu" in base or "deepseek" in base:
        return {"thinking": {"type": "disabled"}}
    return {}


def build_wrappers():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    judge_model = os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "glm-4.7"
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model, api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        temperature=0, timeout=600, max_tokens=100000,  # 实体列举类指标输出长; 上限给足, 按需计费不多收
        extra_body=judge_extra_body()))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=os.environ.get("EVAL_EMB_MODEL", "mlx-community/bge-m3-mlx-fp16"),
        api_key="mlx", base_url=os.environ.get("EVAL_EMB_URL", "http://127.0.0.1:7997/v1"),
        check_embedding_ctx_length=False))  # MLX 服务只收字符串, 不收 token 数组
    return judge, emb


def build_ragas_metrics(judge, emb):
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._context_entities_recall import ContextEntityRecall
    from ragas.metrics._factual_correctness import FactualCorrectness
    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_similarity import SemanticSimilarity, AnswerSimilarity
    from ragas.metrics._nv_metrics import ResponseGroundedness
    return {
        "faithfulness": Faithfulness(llm=judge),
        # strictness=1: ragas 默认让裁判一次生成 n=3 个问题变体, DeepSeek 只支持 n=1
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=emb, strictness=1),
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
        "context_entity_recall": ContextEntityRecall(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge),
        "answer_correctness": AnswerCorrectness(llm=judge, embeddings=emb,
                                                answer_similarity=AnswerSimilarity(embeddings=emb)),
        "semantic_similarity": SemanticSimilarity(embeddings=emb),
        "response_groundedness": ResponseGroundedness(llm=judge),
    }


def string_metrics(response: str, reference: str) -> dict:
    """本地零成本指标: rougeL / bleu / chrf (string_similarity/exact_match 走 ragas 异步通道)。"""
    out = {}
    try:
        from rouge_score import rouge_scorer
        out["rougeL"] = round(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
                              .score(reference, response)["rougeL"].fmeasure, 6)
    except ImportError:
        pass
    try:
        import sacrebleu
        out["bleu"] = round(sacrebleu.sentence_bleu(response, [reference]).score / 100, 6)
        out["chrf"] = round(sacrebleu.sentence_chrf(response, [reference]).score / 100, 6)
    except ImportError:
        pass
    return out


async def rubric_score(dim_desc: str, q: str, ref: str, resp: str) -> float:
    """自定义裁判 1-5 分; JSON 输出, 失败返回 0 并由调用方记录。"""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                         base_url=(os.environ.get("OPENAI_BASE_URL")
                                   or "https://api.openai.com/v1").rstrip("/"), timeout=600)
    model = os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or "glm-4.7"
    r = await client.chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        extra_body=judge_extra_body(),
        messages=[{"role": "user", "content": RUBRIC_PROMPT.format(dim=dim_desc, q=q, ref=ref, resp=resp)}])
    import re
    text = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    return max(1.0, min(5.0, float(d.get("score", 0))))


async def main():
    load_env_file()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    with_noise = "--with-noise" in sys.argv
    limit = 0
    if "--limit" in sys.argv:
        try:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        except (ValueError, IndexError):
            pass

    modes = [m for m in discover_modes() if not args or m in args]
    if not modes:
        print("[eval] 没有可评测的模式 (data/eval_*.jsonl 为空)", flush=True)
        return
    print(f"[eval] 模式={modes} 裁判={os.environ.get('EVAL_JUDGE_MODEL') or os.environ.get('OPENAI_MODEL') or 'glm-4.7'}"
          f" emb={os.environ.get('EVAL_EMB_URL', 'http://127.0.0.1:7997/v1')}", flush=True)

    judge, emb = build_wrappers()
    ragas_ms = build_ragas_metrics(judge, emb)
    from ragas.metrics._string import NonLLMStringSimilarity, ExactMatch
    ragas_ms["string_similarity"] = NonLLMStringSimilarity()   # 本地零成本, 但走 ragas 异步通道
    ragas_ms["exact_match"] = ExactMatch()
    if with_noise:
        from ragas.metrics._noise_sensitivity import NoiseSensitivity
        ragas_ms["noise_sensitivity"] = NoiseSensitivity(llm=judge)

    from ragas.dataset_schema import SingleTurnSample
    cache = load_cache()
    t_all = time.time()

    for mode in modes:
        rows = read_rows(DATA / f"eval_{mode}.jsonl")
        if limit:
            rows = rows[:limit]
        fp = content_fp(mode)
        if not rows or not fp:
            print(f"[eval] {mode}: 无答卷数据, 跳过", flush=True)
            continue
        print(f"[eval] {mode}: {len(rows)} 题", flush=True)
        for r in rows:
            qid, t_q = str(r.get("id")), time.time()
            resp, ref = (r.get("answer") or ""), (r.get("reference") or "")
            sample = SingleTurnSample(
                user_input=r.get("user_input") or "", response=resp,
                retrieved_contexts=r.get("retrieved_contexts") or [],
                reference=ref or None,
                reference_contexts=r.get("reference_contexts") or None)
            done = 0
            # 本地字符串指标 (同步, 免费, rougeL/bleu/chrf)
            if ref:
                for name, val in string_metrics(resp, ref).items():
                    key = f"{mode}@{fp}:{qid}:{name}"
                    if key not in cache:
                        cache[key] = val
                        done += 1
            # ragas 裁判/向量指标 (异步)
            for name, metric in ragas_ms.items():
                key = f"{mode}@{fp}:{qid}:{name}"
                if key in cache:
                    continue
                if name in NEEDS_REFERENCE and not ref:
                    continue
                try:
                    val = await metric.single_turn_ascore(sample, callbacks=[])
                    cache[key] = round(float(val), 6)
                    done += 1
                except Exception as e:  # noqa: BLE001 — 单指标失败不影响其他
                    print(f"[eval]   {mode} q{qid} {name} 失败: {type(e).__name__}: {str(e)[:120]}", flush=True)
            # 自定义 rubric 裁判 (1-5)
            if ref:
                for name, dim in RUBRICS.items():
                    key = f"{mode}@{fp}:{qid}:{name}"
                    if key in cache:
                        continue
                    try:
                        cache[key] = round(await rubric_score(dim, sample.user_input, ref, resp), 2)
                        done += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"[eval]   {mode} q{qid} {name} 失败: {type(e).__name__}: {str(e)[:120]}", flush=True)
            save_cache(cache)
            print(f"[eval]   q{qid} 新算 {done} 项, 累计 {int(time.time()-t_q)}s", flush=True)

        # 汇总该模式
        agg: dict[str, list] = {}
        for key, val in cache.items():
            parts = key.split(":", 2)
            if len(parts) == 3 and parts[0] == f"{mode}@{fp}" and isinstance(val, (int, float)):
                agg.setdefault(parts[2], []).append(val)
        summary = {m: round(statistics.fmean(v), 4) for m, v in sorted(agg.items()) if v}
        (RESULTS / f"scores_{mode}.json").write_text(
            json.dumps({"mode": mode, "n": len(rows), "scores": summary,
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[eval] {mode} 均值: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    # 评测报告 (markdown, 平台 /api/report 读取)
    lines = ["# RAG 评测报告", "", f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for mode in modes:
        p = RESULTS / f"scores_{mode}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        lines += [f"## {mode} (n={s['n']})", "", "| 指标 | 均值 |", "|---|---|"]
        lines += [f"| {m} | {v} |" for m, v in s["scores"].items()]
        lines.append("")
    (RESULTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] 全部完成, 耗时 {int(time.time()-t_all)}s; 缓存 {len(cache)} 条", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
