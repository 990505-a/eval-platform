# -*- coding: utf-8 -*-
"""M2 LLM 基准编排: 调 opencompass 跑常用基准, 支持任意 OpenAI 接口模型 + 限题控费.

用法 (由平台 /api/llm/run 拉起; 也可手动):
    .venv-opencompass\\Scripts\\python.exe run_bench.py <模型名> [基准key]
    基准key: ceval(默认)/cmmlu/mmlu/gpqa/aime2025/mmlu_pro
产物: runs/llm/latest.json  {model: {dataset: {"accuracy": 0-1}}}  (按模型累积合并)
      runs/llm/history.jsonl 逐次运行历史

控费方式: 每次运行前生成"截断数据副本" data/<bench>_lite/ (每子集前 N 题, 对齐 ceval
val 每科 5 题口径), 并复制官方数据集配置为 lite_cfgs/lite_<bench>.py (仅把 path 指向
截断副本), 用官方 --datasets <文件路径> 方式加载 —— 与原生 ceval 流程完全同构。
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parent.parent
WORK_DIR = PLATFORM / "runs" / "llm" / "work"
LATEST = PLATFORM / "runs" / "llm" / "latest.json"
# venv 布局随平台: Windows=Scripts\python.exe + Lib\site-packages; Linux/Mac(容器)=bin/python + lib/python3.x
IS_WIN = os.name == "nt"
if IS_WIN:
    SP = PLATFORM / ".venv-opencompass" / "Lib" / "site-packages"
    OC_VENV_PY = PLATFORM / ".venv-opencompass" / "Scripts" / "python.exe"
else:
    SP = next((PLATFORM / ".venv-opencompass" / "lib").glob("python*/site-packages"))
    OC_VENV_PY = PLATFORM / ".venv-opencompass" / "bin" / "python"
DATA = HERE / "data"
LITE_CFG_DIR = HERE / "lite_cfgs"
# 部分网关套了 Cloudflare, 按特征封 python-urllib 的默认 UA (HTTP 403 error 1010); 统一带浏览器 UA
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}


def chat_completions_url(base_url: str) -> str:
    """拼出完整 chat/completions 端点(纯字符串拼接, 禁止反斜杠).

    opencompass 的 OpenAI 模型默认用 os.path.join 拼 URL, Windows 上会把分隔符换成
    反斜杠(请求路径变成 /v4%5Cchat/completions), 智谱等严格网关直接 404。
    """
    return base_url.rstrip("/") + "/chat/completions"


def preflight(model: str, env: dict) -> str | None:
    """单请求预检: 模型名/密钥/端点是否正确, 通过返回 None, 失败返回可读原因."""
    url = chat_completions_url(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    payload = json.dumps({"model": model,
                          "messages": [{"role": "user", "content": "ping"}],
                          "max_tokens": 16}).encode()  # =1 会卡死 deepseek-v4-flash(服务端 bug)
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", **UA,
        "Authorization": f"Bearer {env.get('OPENAI_API_KEY', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=20):
            return None
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        hint = {401: "密钥认证失败", 403: "无权限(密钥/额度)", 404: "模型名不存在或端点路径不对",
                429: "触发限流", 400: "请求格式被拒(检查模型名)"}.get(e.code, f"HTTP {e.code}")
        return f"{hint}: {body}"
    except Exception as e:  # noqa: BLE001
        return f"连接失败({url}): {type(e).__name__}: {e}"


def negotiate_temperature(model: str, env: dict) -> float:
    """探测网关接受的 temperature: kimi 系网关只允许 1, opencompass 默认发 0.7
    会被 400 全灭(一道题都跑不出). 返回第一个被接受的值, 默认 0.7 不变更既有模型行为."""
    url = chat_completions_url(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    for t in (0.7, 1.0):
        payload = json.dumps({"model": model,
                              "messages": [{"role": "user", "content": "ping"}],
                              "max_tokens": 16, "temperature": t}).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json", **UA,
            "Authorization": f"Bearer {env.get('OPENAI_API_KEY', '')}"})
        try:
            with urllib.request.urlopen(req, timeout=20):
                return t
        except urllib.error.HTTPError as e:
            if "temperature" not in e.read().decode(errors="replace").lower():
                print(f"[bench] temperature={t} 探测异常(HTTP {e.code}), 沿用 {t}", flush=True)
                return t
        except Exception as e:  # noqa: BLE001
            print(f"[bench] temperature={t} 探测失败({type(e).__name__}), 沿用 {t}", flush=True)
            return t
    return 1.0


def negotiate_thinking(model: str, env: dict, want: str) -> dict:
    """思考强度 -> 请求体参数: off=thinking.disabled(智谱/g-bits 系), 低/中/高=reasoning_effort.
    先单请求探测网关是否接受, 被拒(400 等)则回退默认强度, 避免和 temperature 一样全灭."""
    if want not in ("off", "low", "medium", "high"):
        return {}
    body = {"thinking": {"type": "disabled"}} if want == "off" else {"reasoning_effort": want}
    url = chat_completions_url(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    payload = json.dumps({"model": model,
                          "messages": [{"role": "user", "content": "ping"}],
                          "max_tokens": 8, **body}).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", **UA,
        "Authorization": f"Bearer {env.get('OPENAI_API_KEY', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=20):
            return body
    except urllib.error.HTTPError as e:
        print(f"[bench] 思考强度[{want}]被网关拒绝(HTTP {e.code}), 回退默认强度", flush=True)
        return {}
    except Exception as e:  # noqa: BLE001
        print(f"[bench] 思考强度[{want}]探测失败({type(e).__name__}), 回退默认强度", flush=True)
        return {}


def negotiate_max_out(model: str, env: dict, want: int = 100000) -> int:
    """探测网关接受的 max_tokens: 思考模型长推理默认要 10 万, 但 OpenAI 官方严格校验
    (gpt-4o-mini 上限 16384, 超限 400 每题必挂), 国内网关普遍静默放宽。被拒则退 16384。"""
    url = chat_completions_url(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    for mt in (want, 16384):
        payload = json.dumps({"model": model,
                              "messages": [{"role": "user", "content": "ping"}],
                              "max_tokens": mt}).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json", **UA,
            "Authorization": f"Bearer {env.get('OPENAI_API_KEY', '')}"})
        try:
            with urllib.request.urlopen(req, timeout=20):
                return mt
        except urllib.error.HTTPError as e:
            if "max_tokens" not in e.read().decode(errors="replace").lower():
                print(f"[bench] max_tokens={mt} 探测异常(HTTP {e.code}), 沿用 {mt}", flush=True)
                return mt
        except Exception as e:  # noqa: BLE001
            print(f"[bench] max_tokens={mt} 探测失败({type(e).__name__}), 沿用 {mt}", flush=True)
            return mt
    return 16384



def single_instance_guard(model: str) -> None:
    """防双跑: 已有更早启动的同模型 run_bench 时本进程直接退出.

    本机 venv 的 python.exe 是重定向启动器, 每个进程都伴随一个"启动器父进程"镜像,
    故跳过自己的父进程; 再按启动时间兜底, 只保留最早的真实基准进程。
    """
    import psutil
    mine, parent = None, None
    try:
        me = psutil.Process(os.getpid())
        mine, parent = me.create_time(), me.ppid()
    except psutil.NoSuchProcess:
        return
    for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            if p.info["pid"] in (os.getpid(), parent):
                continue
            parts = p.info["cmdline"] or []
            if not parts:
                continue
            # 只认 python 解释器进程, 避免匹配到命令行恰好含相同字符串的 bash/IDE 进程
            exe = str(parts[0]).lower().rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            if exe not in ("python.exe", "python", "pythonw.exe"):
                continue
            cmd = " ".join(str(c) for c in parts)
            if ("run_bench.py" in cmd and model in cmd
                    and p.info["create_time"] < mine):
                print(f"[guard] 检测到更早的同模型基准进程 PID {p.info['pid']} 在跑, "
                      f"本进程退出避免双跑", flush=True)
                sys.exit(0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _progress_watcher(model: str, bench: str, t_start: float, stop: threading.Event) -> None:
    """推理/判卷进行中每 10s 检查最新实验目录的产物, 变化时向主日志写 [progress] 行."""
    total = BENCHES[bench].get("subjects")
    print(f"[progress] 0/{total} 学科 · 等待 opencompass 产出(加载约2分钟)…", flush=True)
    last_pred, last_res = 0, 0
    while not stop.wait(10):
        pred = res = 0
        try:
            dirs = [d for d in WORK_DIR.glob("*/")
                    if d.is_dir() and d.stat().st_mtime >= t_start - 60]
            if dirs:
                cur = max(dirs, key=lambda d: d.stat().st_mtime)
                for sub, holder in (("predictions", "pred"), ("results", "res")):
                    mdir = cur / sub / model
                    if mdir.is_dir():
                        n = len(list(mdir.glob("*.json")))
                        if holder == "pred":
                            pred = n
                        else:
                            res = n
        except OSError:
            pass
        if total and pred >= total:  # 推理完成, 进入判卷阶段
            if res != last_res:
                last_res = res
                print(f"[progress] 判卷 {res}/{total} 学科 ({round(res * 100 / total)}%)", flush=True)
        elif pred != last_pred:
            last_pred = pred
            print(f"[progress] 推理 {pred}/{total} 学科 ({round(pred * 100 / total) if total else 0}%)", flush=True)

# 基准注册表: 官方配置文件(相对 site-packages/opencompass) + 数据形态 + 限题数
#   src      官方配置源文件;  src_path 官方配置里的数据集 path 字符串(替换目标)
#   files    数据形态: csv=按学科文件(截 N 行数据) / jsonl=单文件截 N 行 / json=bbh 任务文件截 N 条
#   est      预计题数(前端展示)
BENCHES = {
    "ceval": None,  # 官方 val 每科 5 题, 本身已是小集, 直接用原生配置
    # src 必须指向自包含的配置文件: 薄封装(mmlu_gen.py 等)复制进 --config-dir 后相对 import 会崩
    "cmmlu": dict(src=r"opencompass/configs/datasets/cmmlu/cmmlu_0shot_cot_gen_305931.py",
                  src_path="opencompass/cmmlu", kind="csv", per=5,
                  label="CMMLU", note="中文综合知识 · 67 学科", est=335, subjects=67),
    "mmlu": dict(src=r"opencompass/configs/datasets/mmlu/mmlu_openai_simple_evals_gen_b618ea.py",
                 src_path="opencompass/mmlu", kind="csv", per=5,
                 extra_cfgs=("mmlu_all_sets.py",),  # 相对 import 的学科清单伴生文件
                 label="MMLU", note="英文综合知识 · 57 学科", est=285, subjects=57),
    # GPQA: diamond 全量 198 题(与公开榜可比的口径), 官方配置原生指向 ./data/gpqa/, 无需截断/替换
    "gpqa": dict(src=r"opencompass/configs/datasets/gpqa/gpqa_gen_4baadb.py",
                 src_path=None, kind="raw", per=198,
                 label="GPQA", note="研究生级理化生 · diamond 全量 198 题", est=198, subjects=1),
    # GSM8K/HellaSwag 已退役(2026-08): 前沿模型 95-99% 已无区分度, 换 AIME 2025 + MMLU-Pro
    "aime2025": dict(src=None, kind="custom", per=30,
                     label="AIME 2025", note="美国数学邀请赛 · I/II 卷全量 30 题 · 整数精确匹配判卷",
                     est=30, subjects=1),
    "mmlu_pro": dict(src=r"opencompass/configs/datasets/mmlu_pro/mmlu_pro_0shot_cot_gen_08c1de.py",
                     src_path="opencompass/mmlu_pro", kind="parquet", per=20,
                     extra_cfgs=("mmlu_pro_categories.py",),
                     label="MMLU-Pro", note="MMLU 继任 · 10 选项研究生级 · 每类前 20 题", est=280, subjects=14),
    # 自有题库: 平台「LLM 基准→自有题库」在线维护/AI 合成采纳, 不截断(kind=custom 走手写 lite_own.py)
    "own": dict(src=None, kind="custom", per=0,
                label="自有题库", note="自己攒的选择题 · LLM 页「自有题库」页签维护", est=0, subjects=1),
}
CEVAL_INFO = dict(label="C-Eval", note="中文综合知识 · 52 学科", est=260,
                  datasets_arg="lite_ceval", subjects=52)
BENCHES["ceval"] = CEVAL_INFO | {"src": None}


def truncate_csv(src: Path, dst: Path, n: int):
    with src.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    keep = [rows[0]] + rows[1:n + 1] if rows else []
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(keep)


def truncate_jsonl(src: Path, dst: Path, n: int):
    with src.open(encoding="utf-8") as f:
        lines = [next(f, None) for _ in range(n)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(l for l in lines if l), encoding="utf-8")


def truncate_json(src: Path, dst: Path, n: int):
    d = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(d, dict) and "examples" in d:
        d["examples"] = d["examples"][:n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def build_lite(bench: str, B: dict) -> Path:
    """生成截断数据副本 + lite 配置文件, 返回 --config-dir 根路径 (幂等).

    opencompass --datasets 按文件名在 <config-dir>/datasets 下匹配,
    故 lite 配置放 lite_cfgs/datasets/lite_<bench>.py, 用 --config-dir 指向 lite_cfgs。
    """
    lite_data = DATA / f"{bench}_lite"
    if B["kind"] == "custom":  # 手写配置(lite_<bench>.py 入库), 数据已就位, 仅校验
        if not (LITE_CFG_DIR / "datasets" / f"lite_{bench}.py").exists():
            raise SystemExit(f"[lite] 缺少手写配置 lite_{bench}.py")
        if not (DATA / bench).is_dir():
            raise SystemExit(f"[lite] 缺少 {DATA / bench}, 请先放入数据集文件")
        return LITE_CFG_DIR
    if B["kind"] == "raw":  # 数据原样使用(如 GPQA diamond 全量), 不做截断
        if not (DATA / bench).is_dir():
            raise SystemExit(f"[lite] 缺少 {DATA / bench}, 请先放入数据集文件")
    elif not lite_data.exists():
        if B["kind"] == "parquet":  # mmlu_pro: 官方 test.parquet 每类截前 N 题, validation 全量保留
            print(f"[lite] 生成截断数据 {lite_data.name} (每类前 {B['per']} 题)", flush=True)
            import pyarrow.parquet as pq
            src_dir = DATA / bench / "data"
            tbl = pq.read_table(src_dir / "test.parquet")
            seen: dict[str, int] = {}
            keep = []
            for i, c in enumerate(tbl.column("category").to_pylist()):
                seen[c] = seen.get(c, 0) + 1
                if seen[c] <= B["per"]:
                    keep.append(i)
            (lite_data / "data").mkdir(parents=True, exist_ok=True)
            pq.write_table(tbl.take(keep), lite_data / "data" / "test.parquet")
            shutil.copy2(src_dir / "validation.parquet", lite_data / "data" / "validation.parquet")
        else:
            print(f"[lite] 生成截断数据 {lite_data.name} (每子集前 {B['per']} 题)", flush=True)
            src_root = {"cmmlu": DATA / "cmmlu", "mmlu": DATA / "mmlu", "bbh": DATA / "BBH",
                        "gsm8k": DATA / "gsm8k", "hellaswag": DATA / "hellaswag"}[bench]
            cut = {"csv": truncate_csv, "jsonl": truncate_jsonl, "json": truncate_json}[B["kind"]]
            # csv: 递归按学科文件; json: bbh 的 data/ 子目录; jsonl: 目录下全部行文件
            glob_pat = "**/*.csv" if B["kind"] == "csv" else ("data/*.json" if bench == "bbh" else "*.jsonl")
            for f in src_root.glob(glob_pat):
                rel = f.relative_to(src_root)
                # gsm8k/hellaswag 的 train 文件只做 few-shot 池, 截 100 行足够
                n = B["per"] if "train" not in f.stem.lower() else min(B["per"], 100)
                cut(f, lite_data / rel, n)
    # lite 配置 = 官方配置逐字复制, 仅替换数据集 path
    (LITE_CFG_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    official = (SP / B["src"]).read_text(encoding="utf-8")
    for extra in B.get("extra_cfgs", ()):  # 官方配置相对 import 的伴生文件, 一并复制
        shutil.copy2(SP / Path(B["src"]).parent / extra, LITE_CFG_DIR / "datasets" / extra)
    if B.get("src_path"):
        lite_path_str = f"./data/{bench}_lite" + ("/data" if bench == "bbh" else "")
        official = official.replace(B["src_path"], lite_path_str)
    kind_zh = "全量" if B["kind"] == "raw" else f"限题版 (每子集前 {B['per']} 题)"
    content = (f"# 动态生成: {B['label']} {kind_zh}. 勿手改\n" + official)
    # 判卷正则兼容全角冒号: 中文模型常把 答案: 写成 答案：, 官方模板只认半角 -> 提不到直接判错
    content = content.replace("答案\\s*:\\s*", "答案\\s*[:：]\\s*")
    content = content.replace("ANSWER\\s*:\\s*", "ANSWER\\s*[:：]\\s*")
    (LITE_CFG_DIR / "datasets" / f"lite_{bench}.py").write_text(content, encoding="utf-8")
    return LITE_CFG_DIR


def ensure_postprocessor_patch() -> None:
    """把 smart_ceval_postprocess 幂等内嵌进 opencompass 的 text_postprocessors 模块。

    mmengine 会把配置里的函数 dump 成 '模块.函数名' 字符串交给判卷子进程解析;
    外部函数(smart_eval.py)在子进程注册表里查不到 -> proc=None -> 判卷全崩(kimi-k3 实测)。
    内嵌后配置写全路径字符串, 与官方 first_capital 走同一条 import 路径; 新装环境自动补打。
    """
    tp = SP / "opencompass" / "utils" / "text_postprocessors.py"
    src = HERE / "smart_eval.py"
    try:
        text = tp.read_text(encoding="utf-8")
    except OSError:
        return
    if not src.exists():
        return
    body = src.read_text(encoding="utf-8")
    marker = "\n\n# ---- eval-platform 内嵌补丁 (源: engines/opencompass/smart_eval.py) ----\n"
    if marker in text:
        head, _, old = text.partition(marker)
        if old.strip() == body.strip():
            return  # 已是最新
        text = head  # 源已更新: 补丁块总在文件末尾, 截掉重嵌
        print("[patch] smart_eval.py 已更新, 重新内嵌补丁", flush=True)
    tp.write_text(text + marker + body + "\n", encoding="utf-8")
    print("[patch] 已内嵌 smart_ceval_postprocess -> opencompass text_postprocessors", flush=True)


def ensure_eval_think_strip_patch() -> None:
    """判卷应用后处理器之前, 先剥离思考链(…</think>最终答案 -> 只看最终段)。

    opencompass 官方后处理器(bbh_freeform 等)按 'answer is' 首次出现提取, 而思考模型的
    prediction 是 推理+</think>+答案, 首次出现落在草稿里 -> 整科 0 分
    (kimi-k3 BBH 实测 navigate/dyck/web_of_lies 等 5 科全灭)。在 _process_predictions
    唯一入口统一剥离, 各基准的后处理器都免修。
    注意: 该文件以 __main__ 收尾且脚本模式顺序执行, helper 必须插到 __main__ 之前,
    追加在文件末尾会导致 NameError(实测踩坑)。幂等 + 自动修复错位旧补丁。
    """
    tp = SP / "opencompass" / "tasks" / "openicl_eval.py"
    try:
        text = tp.read_text(encoding="utf-8")
    except OSError:
        return
    helper = ("# ---- eval-platform 内嵌补丁: 判卷前剥离思考链 ----\n"
              "def _strip_think_for_eval(text):\n"
              '    """thinking 模型 prediction = 推理…</think>最终答案; 后处理器只看最终段。"""\n'
              "    if isinstance(text, str) and '</think>' in text:\n"
              "        tail = text.rsplit('</think>', 1)[1]\n"
              '        if tail.strip():\n'
              '            return tail\n'
              '    return text\n')
    # 1) 包装调用点(4 处预测后处理 + 1 处数据集后处理, 金标准无 think 天然幂等)
    if "proc(_strip_think_for_eval(s), **kwargs)" not in text:
        old = "proc(s, **kwargs)"
        if text.count(old) != 5:
            print(f"[patch] openicl_eval.py 调用点 {text.count(old)} 处 != 5, 跳过思考链剥离补丁", flush=True)
            return
        text = text.replace(old, "proc(_strip_think_for_eval(s), **kwargs)")
    # 2) 移除既有 helper(不管在什么位置), 再插到 __main__ 之前
    text = re.sub(r"\n*# ---- eval-platform 内嵌补丁: 判卷前剥离思考链 ----\ndef _strip_think_for_eval[\s\S]*?return text\n",
                  "\n", text)
    anchor = "if __name__ == '__main__':"
    if anchor in text:
        text = text.replace(anchor, helper + "\n\n" + anchor, 1)
    else:
        text += "\n" + helper
    tp.write_text(text, encoding="utf-8")
    print("[patch] 已内嵌思考链剥离 -> opencompass openicl_eval", flush=True)


def ensure_timeout_patch() -> None:
    """opencompass 的 OpenAI 请求不带超时, 网关静默挂起(不断线也不回数据)时 worker 会
    永远卡死(glm GPQA 实测: 3 个 worker 挂俩, 死等 20+ 分钟)。补 connect/read 超时
    并把 Timeout 纳入既有重试分支。幂等。"""
    tp = SP / "opencompass" / "models" / "openai_api.py"
    try:
        text = tp.read_text(encoding="utf-8")
    except OSError:
        return
    if "timeout=(10, 600)" in text:
        return
    a = "data=json.dumps(data))"          # 非代理分支
    b = "data=json.dumps(data),\n                        proxies=proxies,"  # 代理分支
    c = "except requests.ConnectionError:"
    if not (text.count(a) == 1 and text.count(b) == 1 and text.count(c) == 1):
        print("[patch] openai_api.py 结构变化, 跳过超时补丁", flush=True)
        return
    text = text.replace(a, "data=json.dumps(data), timeout=(10, 600))")
    text = text.replace(b, "data=json.dumps(data),\n                        timeout=(10, 600),\n"
                           "                        proxies=proxies,")
    text = text.replace(c, "except (requests.ConnectionError, requests.Timeout):")
    tp.write_text(text, encoding="utf-8")
    print("[patch] 已内嵌请求超时+重试 -> opencompass openai_api", flush=True)


def ensure_bbh_patch() -> None:
    """BBH 思考模型兼容判卷(幂等, 源变重嵌):
    1) freeform 提取取 'answer is' 最后一次出现 —— 官方取首次, 思考草稿含多次、
       且 <think> 标签前后布局不固定(实测答案段在标签前/后都出现过), 最后一次才是结论;
    2) 评估比较大小写不敏感 —— 官方严格匹配 'no' != 'No' 直接判错。
    freeform 提取在 BBHEvaluator.score 内部, 不走 _process_predictions 的中央剥离,
    故在模块尾部重定义函数与评估器(配置按全路径字符串解析, 尾部重定义生效)。
    """
    tp = SP / "opencompass" / "datasets" / "bbh.py"
    marker = "# ---- eval-platform 内嵌补丁"
    body = (
        "# ---- eval-platform 内嵌补丁: BBH 思考模型兼容判卷 ----\n\n"
        "def bbh_freeform_postprocess(text):\n"
        "    if not text:\n"
        "        return ''\n"
        "    idx = text.lower().rfind('answer is')\n"
        "    if idx >= 0:\n"
        "        text = text[idx + 9:]\n"
        "    ans = text.split('\\n')[0].strip().lstrip(':： ').strip()\n"
        "    if ans.endswith('.'):\n"
        "        ans = ans[:-1].strip()\n"
        "    m = re.search(r'\\*\\*(.*?)\\*\\*', ans)\n"
        "    if m:\n"
        "        ans = m.group(1).strip()\n"
        "    return ans\n\n\n"
        "_BBHBaseEvaluator = BBHEvaluator\n\n\n"
        "class BBHEvaluator(_BBHBaseEvaluator):\n"
        "    \"\"\"归一化比较: 大小写不敏感(no/No), 逗号=空格(列表分隔符差异),\n"
        "    金标准是纯数字时从 pred 提取数字('8 musical instruments' -> '8')。\"\"\"\n"
        "    @staticmethod\n"
        "    def _norm(x):\n"
        "        if not isinstance(x, str):\n"
        "            return x\n"
        "        x = re.sub(r'[,，]\\s*', ' ', x.strip().lower())\n"
        "        return re.sub(r'\\s+', ' ', x).strip()\n\n"
        "    def score(self, predictions, references):\n"
        "        details = []\n"
        "        cnt = 0\n"
        "        for pred, ref in zip(predictions, references):\n"
        "            p = self._norm(bbh_freeform_postprocess(pred))\n"
        "            r = self._norm(ref)\n"
        "            if isinstance(r, str) and re.fullmatch(r'-?\\d+(\\.\\d+)?', r):\n"
        "                nums = re.findall(r'-?\\d+(?:\\.\\d+)?', p if isinstance(p, str) else '')\n"
        "                p = nums[-1] if nums else p\n"
        "            if isinstance(r, str) and r in ('yes', 'no', 'true', 'false'):\n"
        "                p = (p.split() or [''])[0].strip('.,。') if isinstance(p, str) else p\n"
        "            ok = p == r\n"
        "            cnt += ok\n"
        "            details.append({'pred': p, 'answer': r, 'correct': ok})\n"
        "        return {'score': cnt / len(predictions) * 100, 'details': details}\n"
    )
    try:
        text = tp.read_text(encoding="utf-8")
    except OSError:
        return
    if marker in text:
        head, _, old = text.partition(marker)
        if old.strip() == body.strip():
            return  # 已是最新
        text = head  # 补丁块总在文件末尾, 截掉重嵌
    tp.write_text(text + "\n\n" + body, encoding="utf-8")
    print("[patch] 已内嵌 BBH 思考模型兼容补丁", flush=True)


def ensure_aime_patch() -> None:
    """AIME 整数判卷评估器幂等内嵌(math_evaluator.py 尾部)。

    官方 MATHVerifyEvaluator 每题起子进程做符号等价, 子进程 join 超时仅 10s,
    Windows 上加载 sympy 就超 10s -> 30 题全部被判超时算错(deepseek-v4-flash 实测
    accuracy 0.00, details.pred 全是原始推理文本)。AIME 答案恒为 0-999 整数,
    公开榜口径就是整数精确匹配, 故换成免子进程的抽 \\boxed{} 整数比对。
    """
    tp = SP / "opencompass" / "evaluator" / "math_evaluator.py"
    marker = "# ---- eval-platform 内嵌补丁: AIME 整数判卷 ----"
    body = (
        "# ---- eval-platform 内嵌补丁: AIME 整数判卷 ----\n\n"
        "@ICL_EVALUATORS.register_module()\n"
        "class AimeIntEvaluator(BaseEvaluator):\n"
        "    \"\"\"AIME 整数精确匹配(公开榜口径): 抽最后一个 \\\\boxed{} 内容取整数,\n"
        "    无 boxed 则取 </think> 后末段最后一个数字, 与金标整数比对。\"\"\"\n\n"
        "    @staticmethod\n"
        "    def _extract(pred):\n"
        "        import re\n"
        "        if isinstance(pred, (list, tuple)):  # opencompass 管道里预测是['文本']列表\n"
        "            pred = ''.join(str(x) for x in pred)\n"
        "        if not isinstance(pred, str):\n"
        "            return None\n"
        "        seg = pred.rsplit('</think>', 1)[-1]\n"
        "        boxes = re.findall(r'\\\\\\\\boxed\\{([^{}]*)\\}', seg)\n"
        "        cand = boxes[-1] if boxes else seg\n"
        "        nums = re.findall(r'-?\\d+(?:\\.\\d+)?', cand)\n"
        "        if not nums:\n"
        "            return None\n"
        "        try:\n"
        "            return int(float(nums[-1]))\n"
        "        except (TypeError, ValueError):\n"
        "            return None\n\n"
        "    def score(self, predictions, references):\n"
        "        import re\n"
        "        cnt = 0\n"
        "        details = []\n"
        "        for pred, ref in zip(predictions, references):\n"
        "            p = self._extract(pred)\n"
        "            rnums = re.findall(r'-?\\d+(?:\\.\\d+)?', str(ref))\n"
        "            try:\n"
        "                r = int(float(rnums[0])) if rnums else None\n"
        "            except (TypeError, ValueError):\n"
        "                r = None\n"
        "            ok = p is not None and p == r\n"
        "            cnt += ok\n"
        "            details.append({'pred': p, 'answer': r, 'correct': ok})\n"
        "        return {'accuracy': 100 * cnt / max(len(predictions), 1), 'details': details}\n"
    )
    try:
        text = tp.read_text(encoding="utf-8")
    except OSError:
        return
    if marker in text:
        head, _, old = text.partition(marker)
        if old.strip() == body.strip():
            return
        text = head
    tp.write_text(text.rstrip("\n") + "\n\n\n" + body, encoding="utf-8")
    print("[patch] 已内嵌 AIME 整数判卷评估器 -> opencompass math_evaluator", flush=True)


def build_ceval_lite() -> Path:
    """生成 C-Eval 定制配置 (幂等): 官方配置 + 强提示词(只输出字母) + 智能答案提取.

    官方 first_capital_postprocess 取回答的第一个大写字母, 英文作答的模型会被系统性误判
    (glm-5.3 实测 89.2% 的真实分被算成 1.9%), 故统一换成 smart_ceval_postprocess。
    注意必须用全路径字符串形式: 函数对象经 mmengine dump 给判卷子进程后无法解析(会崩)。
    """
    out = LITE_CFG_DIR / "datasets" / "lite_ceval.py"
    official = (SP / "opencompass" / "configs" / "datasets" / "ceval"
                / "ceval_gen_5f30c7.py").read_text(encoding="utf-8")
    content = official
    # 官方源码里 \n 是字符串字面转义(字面反斜杠), 须用 raw string 匹配
    content = content.replace(
        r"\nD. {{D}}\n答案: '",
        r"\nD. {{D}}\n请只输出正确选项的字母(A/B/C/D)，不要输出任何解释。\n答案: '")
    content = content.replace(
        "pred_postprocessor=dict(type=first_capital_postprocess)",
        "pred_postprocessor=dict(type='opencompass.utils.text_postprocessors.smart_ceval_postprocess')")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# 动态生成: C-Eval 定制版 (强提示词 + 智能答案提取). 勿手改\n"
                   + content, encoding="utf-8")
    return LITE_CFG_DIR


CHECK_ENABLED = "--no-check" not in sys.argv
CHECK_ONLY = "--check-only" in sys.argv
sys.argv = [a for a in sys.argv if a not in ("--no-check", "--check-only")]
# --eval-only <实验目录名>: 跳过推理, 复用已有 predictions 只跑判卷+解析 (判卷崩溃后免费重判)
EVAL_ONLY = None
if "--eval-only" in sys.argv:
    _i = sys.argv.index("--eval-only")
    EVAL_ONLY = sys.argv[_i + 1] if _i + 1 < len(sys.argv) else None
    del sys.argv[_i:_i + 2]

# 并发节流: 默认每 worker 1 qps × 1 并发调用 × 2 worker = 2 请求/秒,
# 匹配智谱 coding 类账号的低 QPS 配额(8 worker × batch 8 曾直接打爆账号级限流)。
# 环境变量 OC_QPS / OC_BATCH / OC_WORKERS 可上调。
OC_QPS = float(os.environ.get("OC_QPS", "1"))
OC_BATCH = int(os.environ.get("OC_BATCH", "1"))
OC_WORKERS = int(os.environ.get("OC_WORKERS", "2"))
# 判卷(评分)阶段固定并发: 本地答案比对不花 API 钱, 与推理并发解耦 (OC_EVAL_WORKERS 可调)
OC_EVAL_WORKERS = int(os.environ.get("OC_EVAL_WORKERS", "8"))
# 思考强度: default(不传, 模型自决)/off/low/medium/high —— 平台按模型登记注入, 探测网关后内联
OC_THINK = os.environ.get("OC_THINK", "default").strip().lower()

def _find_exp_dir(t_start: float) -> str | None:
    """本轮启动后最新创建的实验目录名(时间戳), 供判卷阶段 --reuse 复用预测."""
    dirs = [d for d in WORK_DIR.glob("*/")
            if d.is_dir() and d.stat().st_mtime >= t_start - 60]
    return max(dirs, key=lambda d: d.stat().st_mtime).name if dirs else None


MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"
BENCH = sys.argv[2] if len(sys.argv) > 2 else "ceval"
if BENCH not in BENCHES:
    sys.exit(f"[bench] 未知基准 {BENCH}, 可选: {', '.join(BENCHES)}")
B = BENCHES[BENCH]
if BENCH == "own":  # 自有题库: 空题库就地拦截; 实际题数写进 est(平台题级进度分母)
    _own_jsonl = DATA / "own" / "own.jsonl"
    _n_own = sum(1 for _ in _own_jsonl.open(encoding="utf-8")) if _own_jsonl.exists() else 0
    if not _n_own:
        sys.exit("[bench] 自有题库为空: 先在平台 🎓 LLM 基准页「自有题库」添加或采纳题目")
    BENCHES["own"] = B = dict(B, est=_n_own)
    print(f"[bench] 自有题库共 {_n_own} 题", flush=True)

single_instance_guard(MODEL)

# 平台 .env 提供 OPENAI_API_KEY (但平台注入的环境变量优先, 用 setdefault)
env = {k: v for k, v in os.environ.items()}
for line in (PLATFORM / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env.setdefault(k.strip(), v.strip())

# 预检: 模型名/密钥/端点不对时快速失败, 而不是开跑后烧 404 重试 (--no-check 跳过)
_temperature = 0.7  # opencompass 默认值; --no-check 跳过协商时用
_think = {}         # 思考强度探测结果; --no-check 时为空(=默认强度)
_max_out = int(os.environ.get("OC_MAX_OUT", "100000"))  # 输出上限; OpenAI 系需协商降到 16384
# 预检: 模型名/密钥/端点不对时快速失败, 而不是开跑后烧 404 重试 (--no-check 跳过; 仅判卷模式不花 API 也跳过)
if CHECK_ENABLED and not EVAL_ONLY:
    base_url = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    print(f"[preflight] 单请求验证 {MODEL} @ {base_url}", flush=True)
    err = preflight(MODEL, env)
    if err:
        print(f"[preflight] ✗ 预检失败, 已中止(未发起基准任务): {err}", flush=True)
        sys.exit(2)
    print("[preflight] ✓ 模型可达", flush=True)
    # 温度协商: opencompass 默认发 0.7, 只收 1 的网关会全灭; extra_body 最后 update 可覆盖
    _temperature = negotiate_temperature(MODEL, env)
    print(f"[preflight] temperature={_temperature}", flush=True)
    _max_out = negotiate_max_out(MODEL, env, _max_out)
    print(f"[preflight] max_out_len={_max_out}", flush=True)
    _think = negotiate_thinking(MODEL, env, OC_THINK)
    if OC_THINK != "default":
        print(f"[preflight] 思考强度={OC_THINK}" + ("" if _think else " (网关不接受, 回退默认)"), flush=True)
    if CHECK_ONLY:
        print("[preflight] 仅预检(--check-only), 不启动基准", flush=True)
        sys.exit(0)

# 数据集来源: 统一走 lite 配置 (ceval 用定制版: 强提示词 + 智能答案提取, 其余为官方配置截断版)
ensure_postprocessor_patch()
ensure_eval_think_strip_patch()
ensure_bbh_patch()
ensure_timeout_patch()
ensure_aime_patch()
datasets_arg = str(build_ceval_lite() if BENCH == "ceval" else build_lite(BENCH, B))

# 模型动态配置 (沿用既有机制: 写进官方 configs 目录用模式名引用)
# 端点在此处(普通 Python 环境)算好并内联成字面量: mmengine 用懒导入沙箱执行配置,
# 配置内 import os 后立即调用会 RuntimeError; 且该版本 fromfile 默认不替换 {{$VAR}} 模板
CFG_NAME = "dyn_" + re.sub(r"[^0-9a-zA-Z_]", "_", MODEL)
_openai_api_base = chat_completions_url(env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
cfg_content = f'''# -*- coding: utf-8 -*-
"""动态生成: {MODEL}"""
from opencompass.models import OpenAI

api_meta_template = dict(round=[
    dict(role="HUMAN", api_role="HUMAN"),
    dict(role="BOT", api_role="BOT", generate=True),
])

models = [
    dict(
        abbr="{MODEL}",
        type=OpenAI,
        path="{MODEL}",
        key="ENV",
        meta_template=api_meta_template,
        # 显式传完整端点: 绕过 opencompass 默认 os.path.join 在 Windows 上的反斜杠(%5C) 404
        openai_api_base={json.dumps(_openai_api_base)},
        query_per_second={OC_QPS},
        max_out_len={_max_out},
        max_seq_len=128000,
        # extra_body 最后 update 进请求体: 覆盖 opencompass 默认的 temperature=0.7, 附思考强度
        extra_body={json.dumps({"temperature": _temperature, **_think})},
        batch_size={OC_BATCH},
    ),
]
'''
(SP / "opencompass" / "configs" / "models" / "openai" / f"{CFG_NAME}.py").write_text(cfg_content, encoding="utf-8")

# 用 python -m opencompass.cli.main 而不是 Scripts/opencompass.exe 快捷方式:
# 该 shim 在本机会把进程链混入基础解释器(系统 Python)造成重复拉起, 直接跑模块
# 保证整棵进程树都在 .venv-opencompass 解释器内
base_cmd = ([str(OC_VENV_PY), "-m", "opencompass.cli.main", "--config-dir", str(datasets_arg)]
            + ["--datasets", f"lite_{BENCH}", "--models", CFG_NAME, "-w", str(WORK_DIR)])
print(f"[bench] model={MODEL} bench={BENCH}({B['label']}, 预计{B['est']}题) 思考强度={OC_THINK}", flush=True)
print(f"[bench] {' '.join(base_cmd)}", flush=True)
t_start = time.time()
# 实时进度监视: 主日志(平台日志面板)每有产出即出现 [progress] x/y 行
_prog_stop = threading.Event()
_prog_thread = threading.Thread(target=_progress_watcher,
                                args=(MODEL, BENCH, t_start, _prog_stop), daemon=True)
_prog_thread.start()

# 阶段1 推理: 并发由前端填写(OC_WORKERS); 阶段2 判卷: 固定 OC_EVAL_WORKERS, 复用预测不花 API
if EVAL_ONLY:
    exp_ts = EVAL_ONLY
    print(f"[bench] 仅判卷模式: 复用 {exp_ts}, {OC_EVAL_WORKERS} worker", flush=True)
    r = subprocess.run(base_cmd + ["--mode", "eval", "--reuse", exp_ts,
                                   "--max-num-workers", str(OC_EVAL_WORKERS)],
                       env=env, cwd=str(HERE))
else:
    print(f"[bench] 推理阶段: {OC_WORKERS} worker", flush=True)
    r = subprocess.run(base_cmd + ["--mode", "infer", "--max-num-workers", str(OC_WORKERS)],
                       env=env, cwd=str(HERE))
    exp_ts = _find_exp_dir(t_start)
    if r.returncode == 0 and exp_ts:
        print(f"[bench] 判卷阶段: {OC_EVAL_WORKERS} worker (复用 {exp_ts})", flush=True)
        r = subprocess.run(base_cmd + ["--mode", "eval", "--reuse", exp_ts,
                                       "--max-num-workers", str(OC_EVAL_WORKERS)],
                           env=env, cwd=str(HERE))
    elif r.returncode != 0:
        print(f"[bench] 推理阶段退出码 {r.returncode}, 跳过判卷", flush=True)
_prog_stop.set()
_prog_thread.join(timeout=2)
if r.returncode != 0:
    print(f"[bench] opencompass 退出码 {r.returncode}, 继续尝试解析已产出的结果", flush=True)

# ---- 解析: work_dir/<时间戳>/results/<model>/<dataset>.json ----
# 只认本轮启动之后产出的结果目录, 防止失败时误读历史成绩张冠李戴
model_result: dict = {}
if WORK_DIR.exists():
    for run_dir in sorted([p for p in WORK_DIR.glob("*/results") if p.is_dir()], reverse=True):
        if EVAL_ONLY:
            # 仅判卷模式: 结果覆写不刷新 results 目录 mtime, 按目录名匹配而不是时间
            if run_dir.parent.name != EVAL_ONLY:
                continue
        elif run_dir.stat().st_mtime < t_start - 60:
            continue
        for mf in run_dir.glob(f"{MODEL}/*.json"):
            try:
                metric = json.loads(mf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            acc = metric.get("accuracy", metric.get("score"))  # 各基准评估器键名不一: ceval=accuracy, bbh=score
            if acc is None:
                continue
            if acc > 1:  # opencompass 输出 0-100, 归一到 0-1 供前端展示
                acc = acc / 100
            model_result[mf.stem] = {"accuracy": round(acc, 4)}
        if model_result:
            break

if not model_result:
    sys.exit(f"[bench] 未解析到 {MODEL} 的成绩, 查看 runs/llm/work 与 runs/llm/log.txt")

# 合并进已有矩阵 (多模型/多基准累积对比; 同模型重跑同基准则覆盖该基准部分)
matrix = {}
if LATEST.exists():
    try:
        matrix = json.loads(LATEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        matrix = {}
matrix.setdefault(MODEL, {}).update(model_result)
LATEST.parent.mkdir(parents=True, exist_ok=True)
LATEST.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[bench] {MODEL} x {B['label']}: {len(model_result)} 项, 成绩已合并 -> {LATEST}", flush=True)

# 运行历史 (平台 /api/llm/history 展示)
accs = [v["accuracy"] for v in model_result.values()]
hist = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODEL, "bench": B["label"],
        "n": len(model_result), "mean": round(sum(accs) / len(accs), 4)}
hp = PLATFORM / "runs" / "llm" / "history.jsonl"
hp.parent.mkdir(parents=True, exist_ok=True)
with open(hp, "a", encoding="utf-8") as f:
    f.write(json.dumps(hist, ensure_ascii=False) + "\n")
print(f"[bench] 历史已记录 -> {hp}", flush=True)

# 每轮归档(平台"历史轮次"面板查看/删除); 含各子集分数明细
arch = PLATFORM / "runs" / "llm" / "archive" / (time.strftime("%Y%m%d-%H%M%S") + ".json")
arch.parent.mkdir(parents=True, exist_ok=True)
arch.write_text(json.dumps({**hist, "scores": model_result}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[bench] 归档 -> {arch}", flush=True)
