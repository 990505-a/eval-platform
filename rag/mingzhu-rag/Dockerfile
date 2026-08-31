# 名著RAG (LightRAG): OpenAI 兼容 LLM + Embedding, 文件存储(JSON + NanoVectorDB + NetworkX)
FROM python:3.12-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# [api] extra 带出 lightrag-server 及其 WebUI/依赖; httpx 供 ingest.py 用
# (包名已从 lightrag-hi 改为 lightrag-hku)
RUN pip install --no-cache-dir "lightrag-hku[api]" httpx

WORKDIR /app
COPY ingest.py .
COPY corpus ./corpus

# 服务端配置全部走环境变量 (见 docker-compose.yml rag-mingzhu 服务):
#   LLM_BINDING/LLM_BINDING_HOST/LLM_BINDING_API_KEY/LLM_MODEL
#   EMBEDDING_BINDING/EMBEDDING_BINDING_HOST/EMBEDDING_BINDING_API_KEY/EMBEDDING_MODEL
#   WORKING_DIR=./rag_storage (挂载持久化), HOST/PORT=0.0.0.0:9621
ENV HOST=0.0.0.0 PORT=9621 WORKING_DIR=/app/rag_storage
EXPOSE 9621

CMD ["lightrag-server"]
