# -*- coding: utf-8 -*-
"""名著语料入库脚本 —— 把 corpus/ 下文本分片推给运行中的 lightrag-server。

用法 (容器内):
    python ingest.py                          # 全部语料入库
    python ingest.py --books 红楼梦 西游记      # 指定书目
    python ingest.py --pieces 2               # 每本书只入前 2 片 (小额冒烟测试)
    python ingest.py --dry-run                # 只统计不上传

成本提示: 入库会对每个 chunk 做 embedding + LLM 实体抽取建知识图谱,
全书全量入库约 1.4 万+ chunk, 会产生真实 API 费用; 建议先用 --pieces 验证再全量。
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx

RAG_URL = os.environ.get("RAG_URL", "http://127.0.0.1:9621").rstrip("/")

CORPUS = Path(__file__).parent / "corpus"
DEFAULT_PIECE_CHARS = 150_000  # 单次上传字符数; 服务端再按 CHUNK_SIZE(1200 token) 细分


def clean_gutenberg(text: str) -> str:
    """去掉莎士比亚全集(Gutenberg 版)头部许可声明与尾部说明。"""
    m = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if m:
        text = text[:m.start()]
    return text.strip()


def pieces_of(text: str, piece_chars: int):
    for i in range(0, len(text), piece_chars):
        yield text[i:i + piece_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", default=None, help="书目名(不带 .txt), 缺省全部")
    ap.add_argument("--pieces", type=int, default=None, help="每本书最多上传的分片数")
    ap.add_argument("--piece-chars", type=int, default=DEFAULT_PIECE_CHARS,
                    help=f"每片字符数(默认 {DEFAULT_PIECE_CHARS}; 冒烟可用小值如 20000)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(CORPUS.glob("*.txt"))
    if args.books:
        files = [f for f in files if f.stem in args.books]
    if not files:
        sys.exit(f"corpus/ 下没有可入库的文本: {CORPUS}")
    if not args.dry_run:
        r = httpx.get(f"{RAG_URL}/health", timeout=10)
        r.raise_for_status()
        print(f"[ok] lightrag-server 可达: {RAG_URL}")

    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if "GUTENBERG" in text[:2000]:
            text = clean_gutenberg(text)
        n = len(list(pieces_of(text, args.piece_chars)))
        total += n
        print(f"[corpus] {f.name}: {len(text):,} 字 → {n} 片 (每片 {args.piece_chars:,})")
        if args.dry_run:
            continue
        for idx, piece in enumerate(pieces_of(text, args.piece_chars), 1):
            if args.pieces and idx > args.pieces:
                break
            resp = httpx.post(f"{RAG_URL}/documents/texts",
                              json={"texts": [piece],
                                    "file_sources": [f"{f.name}#p{idx}"]},
                              timeout=300)
            resp.raise_for_status()
            print(f"  [{f.name}#{idx}] 已入队: {resp.json()}", flush=True)
            time.sleep(1)  # 温和一点的提交节奏, 避免压垮服务端任务队列
    print(f"[done] 共 {total} 片{' (dry-run 未上传)' if args.dry_run else ''}。"
          f"入库进度见 http://localhost:9621/webui 的 Documents 页。")


if __name__ == "__main__":
    main()
