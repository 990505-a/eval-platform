# 名著智能体 (deepagents): python:3.12 满足 deepagents >=3.11 要求
FROM python:3.12-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# deepagents 不自带 openai 接入, 需显式装 langchain-openai; langfuse 用于本地轨迹上报
RUN pip install --no-cache-dir deepagents langchain-openai \
        fastapi "uvicorn[standard]" httpx langfuse

WORKDIR /app
COPY server.py tasks.jsonl ./

ENV AGENT_PORT=8820
EXPOSE 8820

CMD ["python", "server.py"]
