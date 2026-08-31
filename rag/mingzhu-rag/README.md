# 名著RAG (LightRAG)

数据源: 四大名著(红楼梦/西游记/水浒传/三国演义) + 莎士比亚全集, 见 `corpus/`。

## 启动

随主 compose 启动: `docker compose up -d rag-mingzhu` → http://localhost:9621 (WebUI /webui)

**embedding / rerank 已本地化**：默认走宿主机 MLX 服务（`../mlx-models/`，Metal GPU，
bge-m3 + Qwen3-Reranker-0.6B，约 1.5GB 内存），不消耗 API 额度。LLM（建图/查询）仍走云端
`.env` 的 `OPENAI_*`。要用云端 embedding/rerank 时在 `.env` 覆盖
`EMBEDDING_BINDING_HOST`/`EMBEDDING_MODEL`/`RERANK_*` 后 `docker compose up -d rag-mingzhu`。

## 入库(会产生真实 API 费用: 每片做 embedding + LLM 实体抽取建图)

```
docker compose exec rag-mingzhu python ingest.py --dry-run     # 只统计
docker compose exec rag-mingzhu python ingest.py --pieces 2    # 每本书 2 片冒烟
docker compose exec rag-mingzhu python ingest.py               # 全量(约 100 片)
docker compose exec rag-mingzhu python ingest.py --books 红楼梦 # 指定书目
```

索引落在宿主机 `rag/mingzhu-rag/rag_storage/`（compose 挂载），重建容器不丢。

## 查询

- WebUI: http://localhost:9621/webui (可选 naive/local/global/hybrid/mix 模式)
- REST: `POST /query {"query": "...", "mode": "mix"}`
- 名著智能体(agents/mingzhu-agent)的工具 `search_classics` 就是调这个接口。
