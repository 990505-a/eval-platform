# -*- coding: utf-8 -*-
r"""检索胶水脚本: 题集 × 后端模式 → data/eval_<mode>.jsonl (+ .meta.json 参数指纹)

后端切换 (RAG_BACKEND env, 平台 ⚙️/检索参数面板可改):
  lightrag — LightRAG 服务(/query, 知识图谱 5 模式 + bypass 裸LLM对照)
  ragflow  — RAGFlow /api/v1/retrieval (向量/加权/重排 3 模式; 答案由胶水用 OPENAI_* 生成)

模式清单:
  lightrag: naive local global hybrid mix bypass
  ragflow:  rf-vector rf-hybrid rf-rerank

产出契约 (与平台 app.py 对齐, 平台零侵入读取):
  data/eval_<mode>.jsonl      每行 {id, book, user_input, answer, retrieved_contexts,
                               reference, reference_contexts?, backend, mode, latency_ms}
  data/eval_<mode>.meta.json  {"fp": 参数指纹} — 与平台 _param_fp 同算法, 改参数/题集即失效

环境变量(平台注入; 单独手跑有默认值):
  RAG_BACKEND / RAG_PARAMS(json) / RAG_LIGHT_URL / RAGFLOW_URL / RAGFLOW_DATASET
  RAGFLOW_EMAIL / RAGFLOW_PASSWORD / RAGFLOW_RERANK_ID / OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
"""
import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # engines/ragas
DATA = ROOT / "data"

# ---- 平台公共检索参数默认值(须与 app.py RETRIEVAL_DEFAULTS 一致, 指纹算法依赖) ----
RETRIEVAL_DEFAULTS = {"top_k": 40, "chunk_top_k": 20, "max_total_tokens": 8000,
                      "enable_rerank": False, "response_type": "Multiple Paragraphs",
                      "similarity_threshold": 0.2}

BACKEND = os.environ.get("RAG_BACKEND", "ragflow").strip().lower()
MODES = {
    "lightrag": ["naive", "local", "global", "hybrid", "mix", "bypass"],
    "ragflow": ["rf-vector", "rf-hybrid", "rf-rerank", "rf-term", "rf-keyword"],
}[BACKEND]

LIGHT_URL = os.environ.get("RAG_LIGHT_URL", "http://127.0.0.1:9621").rstrip("/")
RF_URL = os.environ.get("RAGFLOW_URL", "http://127.0.0.1:8180").rstrip("/")
RF_DATASET = os.environ.get("RAGFLOW_DATASET", "mingzhu-test")   # 名字或 id
RF_EMAIL = os.environ.get("RAGFLOW_EMAIL", "eval@mingzhu.local")
RF_PASSWORD = os.environ.get("RAGFLOW_PASSWORD", "Eval12345!")
# 模型引用用三段式(模型@实例@provider): 多实例共存时两段式会按"default"实例解析失败
RF_RERANK_ID = os.environ.get("RAGFLOW_RERANK_ID",
                              "mlx-community/Qwen3-Reranker-0.6B-4bit@mlx-local@OpenAI-API-Compatible")
ANSWER_MODEL = os.environ.get("OPENAI_MODEL", "glm-4.7")


def load_env_file():
    """独立手跑时读平台 .env (平台拉起时已注入环境变量, setdefault 不覆盖)。"""
    envf = ROOT.parent.parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def load_params() -> dict:
    raw = os.environ.get("RAG_PARAMS")
    out = dict(RETRIEVAL_DEFAULTS)
    if raw:
        try:
            out.update({k: v for k, v in json.loads(raw).items() if k in RETRIEVAL_DEFAULTS})
        except json.JSONDecodeError:
            print("[retrieve] RAG_PARAMS 解析失败, 用默认参数", flush=True)
    return out


def param_fp(params: dict, qfile: Path) -> str:
    """与平台 app.py._param_fp 逐字对齐: sorted k=v 串 + 题集 md5 再 md5。"""
    s = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    if qfile.exists():
        s += "|qset:" + hashlib.md5(qfile.read_bytes()).hexdigest()[:8]
    return hashlib.md5(s.encode()).hexdigest()[:8]


def read_questions() -> list[dict]:
    qf = DATA / "eval_questions.jsonl"
    rows = []
    for line in qf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def read_rows_q(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ---------------- 答案生成 (ragflow 模式与 bypass 共用; LightRAG 模式用其自身 /query 生成) ----------------
def llm_answer(question: str, contexts: list[str]) -> tuple[str, int]:
    """contexts 为空 = bypass 裸答。返回 (答案, prompt_tokens估算)。"""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                    base_url=(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
                    timeout=600)
    if contexts:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (f"你是名著知识助手。请仅依据下面的资料回答问题，条理清晰，"
                  f"并在结尾标注依据的资料编号（如 [1][2]）。资料不足以回答的部分请明确说明。\n\n"
                  f"【资料】\n{ctx}\n\n【问题】{question}")
    else:
        prompt = f"你是名著知识助手，请凭自己对中国古典名著的了解回答：{question}"
    r = client.chat.completions.create(model=ANSWER_MODEL, messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip(), len(prompt)


# ---------------- LightRAG 后端 ----------------
def query_lightrag(question: str, mode: str, params: dict, only_context: bool) -> tuple[str, list[str]]:
    """一次 /query 调用。only_context=True 时返回 (原始上下文文本, 解析出的chunks)。"""
    body = {"query": question, "mode": mode, "top_k": int(params["chunk_top_k"]),
            "only_need_context": only_context,
            "response_type": params.get("response_type", "Multiple Paragraphs")}
    r = httpx.post(f"{LIGHT_URL}/query", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    text = d.get("response") or d.get("data") or ""
    return text, parse_lightrag_chunks(text)


def parse_lightrag_chunks(text: str) -> list[str]:
    """only_need_context 的响应里 chunks 以 ```json 围栏/内嵌 JSON 对象给出, 稳健提取 content。"""
    chunks = []
    for m in re.finditer(r"```(?:json)?\s*(.+?)```", text, re.DOTALL):
        frag = m.group(1).strip()
        try:
            obj = json.loads(frag)
            for item in (obj if isinstance(obj, list) else [obj]):
                c = item.get("content") if isinstance(item, dict) else None
                if c:
                    chunks.append(c)
        except json.JSONDecodeError:
            pass
    if not chunks:  # 围栏解析不出 → 逐对象正则兜底
        for m in re.finditer(r'\{\s*"reference_id"[^{}]*?"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
            try:
                chunks.append(json.loads(f'"{m.group(1)}"'))
            except json.JSONDecodeError:
                pass
    if not chunks and text.strip():  # 最后兜底: 整段当一个 context (KG 模式的实体/关系段)
        chunks = [text.strip()[:4000]]
    return chunks[:int(RETRIEVAL_DEFAULTS["chunk_top_k"]) or 20]


def eval_lightrag(question: str, mode: str, params: dict) -> tuple[str, list[str], int]:
    t0 = time.time()
    if mode == "bypass":
        ans, _ = llm_answer(question, [])
        return ans, [], int(time.time() - t0)
    _, ctxs = query_lightrag(question, mode, params, only_context=True)   # 先取上下文
    ans, _ = query_lightrag(question, mode, params, only_context=False)   # 再生成答案
    return ans, ctxs, int(time.time() - t0)


# ---------------- RAGFlow 后端 ----------------
# ragflow 镜像内置 RSA 公钥(各安装同一把), 登录密码 = RSA(base64(明文)) 再 base64 —— 与 api/utils/crypt.py 同逻辑
RF_PUBLIC_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArq9XTUSeYr2+N1h3Afl/
z8Dse/2yD0ZGrKwx+EEEcdsBLca9Ynmx3nIB5obmLlSfmskLpBo0UACBmB5rEjBp
2Q2f3AG3Hjd4B+gNCG6BDaawuDlgANIhGnaTLrIqWrrcm4EMzJOnAOI1fgzJRsOO
UEfaS318Eq9OVO3apEyCCt0lOQK6PuksduOjVxtltDav+guVAA068NrPYmRNabVK
RNLJpL8w4D44sfth5RvZ3q9t+6RTArpEtc5sh5ChzvqPOzKGMXW83C95TxmXqpbK
6olN4RevSfVjEAgCydH6HN6OhtOQEcnrU97r9H0iZOWwbw3pVrZiUkuRD1R56Wzs
2wIDAQAB
-----END PUBLIC KEY-----"""


def rf_crypt(password: str) -> str:
    from Cryptodome.PublicKey import RSA
    from Cryptodome.Cipher import PKCS1_v1_5
    key = RSA.importKey(RF_PUBLIC_PEM, "Welcome")
    b64 = base64.b64encode(password.encode("utf-8"))
    return base64.b64encode(PKCS1_v1_5.new(key).encrypt(b64)).decode()


class RagFlow:
    """ragflow /api/v1 极简客户端: 登录换 token + 按名解析 dataset + retrieval。"""

    def __init__(self):
        self.token = None
        self.dataset_id = None

    def _login(self):
        r = httpx.post(f"{RF_URL}/api/v1/auth/login",
                       json={"email": RF_EMAIL, "password": rf_crypt(RF_PASSWORD)}, timeout=60)
        r.raise_for_status()
        tok = r.headers.get("Authorization") or r.headers.get("authorization")
        if not tok:
            body = r.json().get("data") or {}
            tok = body.get("access_token")
        if not tok:
            raise RuntimeError(f"ragflow 登录失败: {r.text[:200]}")
        self.token = tok.split(" ", 1)[-1].strip()

    def _get(self, path, **kw):
        r = httpx.get(f"{RF_URL}{path}", headers={"Authorization": self.token},
                      params=kw, timeout=120)
        r.raise_for_status()
        return r.json()

    def ensure_dataset(self):
        if len(RF_DATASET) == 32 and not RF_DATASET.startswith("mingzhu"):
            self.dataset_id = RF_DATASET
            return
        d = self._get("/api/v1/datasets", name=RF_DATASET, page=1)
        rows = d.get("data") or []
        if not rows:
            raise RuntimeError(f"ragflow 里找不到数据集 {RF_DATASET}")
        self.dataset_id = rows[0]["id"]

    def retrieval(self, question: str, mode: str, params: dict) -> list[dict]:
        body = {"question": question, "dataset_ids": [self.dataset_id],
                "page_size": int(params["chunk_top_k"]),
                "similarity_threshold": float(params["similarity_threshold"])}
        if mode == "rf-vector":
            body["vector_similarity_weight"] = 1.0        # 纯向量
        elif mode == "rf-hybrid":
            body["vector_similarity_weight"] = 0.5        # 向量+词项加权
        elif mode == "rf-rerank":
            body["vector_similarity_weight"] = 0.5
            body["rerank_id"] = RF_RERANK_ID
        elif mode == "rf-term":
            body["vector_similarity_weight"] = 0.0        # 纯词项(BM25 系字面命中)
        elif mode == "rf-keyword":
            body["vector_similarity_weight"] = 0.5        # 加权 + LLM 抽关键词改写查询(需默认 chat model)
            body["keyword"] = True
        for attempt in (1, 2):  # ragflow 会话 token 有时效, 401 时重登一次再试
            r = httpx.post(f"{RF_URL}/api/v1/retrieval", headers={"Authorization": self.token},
                           json=body, timeout=300)
            if r.status_code == 401 and attempt == 1:
                print("[retrieve] ragflow token 过期, 重新登录", flush=True)
                self._login()
                continue
            r.raise_for_status()
            d = r.json()
            if d.get("code") != 0:
                raise RuntimeError(f"ragflow retrieval 失败: {d.get('message')}")
            return (d.get("data") or {}).get("chunks") or []
        raise RuntimeError("ragflow retrieval 两次尝试均失败")


def eval_ragflow(question: str, mode: str, params: dict, rf: RagFlow) -> tuple[str, list[str], int]:
    t0 = time.time()
    chunks = rf.retrieval(question, mode, params)
    ctxs = [c.get("content") or "" for c in chunks if c.get("content")]
    ans, _ = llm_answer(question, ctxs)
    return ans, ctxs, int(time.time() - t0)


# ---------------- 主流程 ----------------
def main():
    load_env_file()
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    modes = [m for m in MODES if not only or m in only]
    if not modes:
        print(f"[retrieve] 无可跑模式: 请求 {only}, 后端 {BACKEND} 支持 {MODES}", flush=True)
        return
    params = load_params()
    questions = read_questions()
    print(f"[retrieve] 后端={BACKEND} 模式={modes} 题数={len(questions)} 参数={params}", flush=True)

    rf = None
    if BACKEND == "ragflow":
        rf = RagFlow()
        rf._login()
        rf.ensure_dataset()
        print(f"[retrieve] ragflow 登录成功, dataset={rf.dataset_id}", flush=True)
    else:
        probe = httpx.get(f"{LIGHT_URL}/health", timeout=10)
        print(f"[retrieve] lightrag health={probe.status_code}", flush=True)

    fp = param_fp(params, DATA / "eval_questions.jsonl")
    for mode in modes:
        out, meta = DATA / f"eval_{mode}.jsonl", DATA / f"eval_{mode}.meta.json"
        # 模式级断点续跑: 同指纹且题数齐全的答卷直接跳过, 中断后重跑只补缺的模式
        if out.exists() and meta.exists():
            try:
                if json.loads(meta.read_text(encoding="utf-8")).get("fp") == fp:
                    done_rows = read_rows_q(out)
                    if len(done_rows) >= len(questions):
                        print(f"[retrieve] {mode}: 已有同指纹完整答卷({len(done_rows)}题), 跳过", flush=True)
                        continue
            except (json.JSONDecodeError, OSError):
                pass
        rows, t_mode = [], time.time()
        for q in questions:
            try:
                if BACKEND == "ragflow":
                    ans, ctxs, lat = eval_ragflow(q["user_input"], mode, params, rf)
                else:
                    ans, ctxs, lat = eval_lightrag(q["user_input"], mode, params)
                row = {"id": q["id"], "book": q.get("book", "?"), "user_input": q["user_input"],
                       "answer": ans, "retrieved_contexts": ctxs, "reference": q.get("reference", ""),
                       "backend": BACKEND, "mode": mode, "latency_ms": lat * 1000}
                if q.get("reference_contexts"):
                    row["reference_contexts"] = q["reference_contexts"]
                rows.append(row)
                print(f"[retrieve] {mode} q{q['id']} ok ctx={len(ctxs)} {lat}s", flush=True)
            except Exception as e:  # noqa: BLE001 — 单题失败不终止整轮
                print(f"[retrieve] {mode} q{q.get('id')} 失败: {type(e).__name__}: {e}", flush=True)
        tmp = out.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        tmp.replace(out)
        meta.write_text(json.dumps({"fp": fp, "backend": BACKEND,
                                    "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False),
                       encoding="utf-8")
        print(f"[retrieve] {mode} 完成: {len(rows)}/{len(questions)} 题, 耗时 {int(time.time()-t_mode)}s -> {out.name}",
              flush=True)


if __name__ == "__main__":
    main()
