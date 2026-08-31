# RAGFlow (对比 RAG)

与 LightRAG (`../mingzhu-rag/`) 平行的第二套 RAG，用于对比评测。
**同一套 embedding(bge-m3) + rerank(Qwen3-Reranker) —— 都走宿主机 MLX 服务**，同语料、同切块参数。

## 架构

- 栈：`ragflow(v0.27.0) + elasticsearch(8.11.3, 3GB) + mysql + minio + valkey(redis)`
- Web：http://localhost:8180 （账号 `eval@mingzhu.local` / `Eval12345!`）
- 启动：`docker compose -f rag/ragflow/docker-compose.yml -p eval-platform-ragflow up -d`
- 停止：`... down`（数据在命名卷）

## 已配置(自动化完成, 重装不必重复)

1. 模型 provider `OpenAI-API-Compatible` 实例 `mlx-local` → `http://host.docker.internal:7997/v1`
   - embedding: `mlx-community/bge-m3-mlx-fp16`，rerank: `mlx-community/Qwen3-Reranker-0.6B-4bit`
2. 数据集 `mingzhu-test`（chunk_token_num=1200 对齐 LightRAG），已入库红楼梦首片 20000 字符 → 18 chunks
3. v0.27 的新 REST API 前缀是 `/api/v1`；密码传输为 RSA 加密（`docker exec <容器> python3 -c "from api.utils.crypt import crypt; print(crypt('密码'))"`）

## 对比要点(同题: 甄士隐与贾雨村的关系, 同 18 chunks 语料)

| | LightRAG (9621) | RAGFlow (8180) |
|---|---|---|
| 检索方式 | 向量 + **知识图谱**(GLM 抽取实体/关系) + 重排 | 向量(ES) + 可选关键词混合 + 重排 |
| 检索返回 | KG 实体描述 + 原文片段 | 纯原文片段 + 相似度分 |
| 查询延迟 | ~140s (GLM 思考型关键词抽取) | 秒级 (无 LLM 依赖; keyword=true 才用) |
| 入库成本 | 高 (每 chunk 一次 GLM 实体抽取 ~3min) | 低 (只做切块+本地 embedding, 秒级) |
| WebUI | /webui 简单查询 | 完整产品化 (知识库/聊天/Agent 编排) |

## 检索 API

```bash
TOKEN=$(curl -s -D- -o /dev/null -X POST http://localhost:8180/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"eval@mingzhu.local\",\"password\":\"$(docker exec eval-platform-ragflow-ragflow-1 python3 -c "from api.utils.crypt import crypt; print(crypt('Eval12345!'))")\"}" \
  | grep -i ^authorization | awk '{print $2}' | tr -d '\r')

curl -X POST http://localhost:8180/api/v1/retrieval -H "Authorization: $TOKEN" \
  -H 'Content-Type: application/json' -d '{
    "question": "甄士隐是谁？",
    "dataset_ids": ["<dataset_id>"],
    "page_size": 5,
    "rerank_id": "mlx-community/Qwen3-Reranker-0.6B-4bit@OpenAI-API-Compatible"
  }'
```

## 踩坑记录

- v0.27 无 slim 镜像（只有 full 3.4GB）；`API_PROXY_SCHEME=python` 必须显式设置，否则 python/go server 分支都不启动（只有 nginx 起来，全程 502）
- Docker Desktop 注入的 http_proxy 会让容器访问宿主机 MLX 变 502，已用空 proxy + no_proxy 覆盖
- 上传文本切片要按**字符**切，按字节切会截断多字节汉字导致解析出 1 个废 chunk
