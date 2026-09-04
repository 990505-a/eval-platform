# 测评聚合平台 (FastAPI 平台 + OpenCompass LLM 基准 + promptfoo 红队)
# 构建:  docker compose build   (已配置国内镜像源直连, 并在 compose 里清空代理注入)
# 启动:  docker compose up -d   (见 docker-compose.yml, 首次需先配置 .env)

FROM python:3.12-slim

# apt 换清华镜像源 (Debian trixie 的 deb822 格式源文件), 走 https + 8 次重试
# (并发拉包时部分连接会被网络抖掉, 实测 http:80 直连偶发 Unable to connect)
# build-essential/git: opencompass 依赖编译; nodejs/npm: 红队模块 promptfoo
RUN sed -i -e 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
           -e 's|http://mirrors.tuna|https://mirrors.tuna|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=8 update \
    && apt-get -o Acquire::Retries=8 install -y --no-install-recommends \
        build-essential git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip 统一走清华 PyPI 镜像 (平台依赖 + opencompass venv)
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 平台本体运行时依赖 (app.py)
RUN pip install --no-cache-dir fastapi uvicorn httpx psutil python-multipart

# LLM 基准模块: opencompass 独立 venv (对齐 DEPLOY.md 的 .venv-opencompass 布局)
RUN python -m venv .venv-opencompass \
    && .venv-opencompass/bin/pip install --no-cache-dir opencompass==0.5.3 psutil

# RAG 测评模块: ragas 胶水层独立 venv (双后端 lightrag/ragflow, 见 engines/ragas)
COPY engines/ragas/requirements.txt engines/ragas/requirements.txt
RUN python -m venv .venv-ragas \
    && .venv-ragas/bin/pip install --no-cache-dir -r engines/ragas/requirements.txt

# 智能体测评模块: DeepEval 独立 venv (engines/deepeval, 双口径指标 + Langfuse 抽样评分)
COPY engines/deepeval/requirements.txt engines/deepeval/requirements.txt
RUN python -m venv .venv-agenteval \
    && .venv-agenteval/bin/pip install --no-cache-dir -r engines/deepeval/requirements.txt

# 智能体基准模块: Harbor (Terminal-Bench 2 官方 harness; 任务容器经挂载的 docker.sock 由宿主机 daemon 起)
RUN python -m venv .venv-tbench \
    && .venv-tbench/bin/pip install --no-cache-dir harbor \
    # 平台补丁: Terminus-2 装工具超时写死 120s(pip 装 torch 必超时), 放宽到 600s
    && sed -i 's/_TOOL_INSTALL_TIMEOUT_SEC = 120$/_TOOL_INSTALL_TIMEOUT_SEC = 600/' \
        .venv-tbench/lib/python3.12/site-packages/harbor/agents/terminus_2/tmux_session.py

# gyp: better-sqlite3(promptfoo 依赖)在 linux/arm64 无预编译二进制需源码编译,
# 而 Debian 的 node-gyp 依赖独立的 gyp 包 (单独成层, 不作废上面 2GB 的 opencompass 缓存层)
RUN apt-get -o Acquire::Retries=8 update && apt-get -o Acquire::Retries=8 install -y --no-install-recommends gyp \
    && rm -rf /var/lib/apt/lists/*

# 官方 Node 20 覆盖 Debian node (Debian 拆包的 node-gyp 缺 gyp 模块, 编译原生依赖必挂;
# 官方发行版自带完整 node-gyp/gyp-next). node 头文件下载同样走 npmmirror
ARG TARGETARCH
RUN curl -fsSL https://registry.npmmirror.com/-/binary/node/v20.20.2/node-v20.20.2-linux-${TARGETARCH}.tar.xz \
        -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && node --version && npm --version
ENV npm_config_disturl=https://registry.npmmirror.com/-/binary/node

# 红队模块: promptfoo (node_modules/.bin/promptfoo)
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund --registry=https://registry.npmmirror.com

# docker CLI + compose 插件: harbor(智能体基准)在容器内经挂载的 /var/run/docker.sock 操作宿主机 daemon,
# 它 shell 出 docker/docker compose 命令 → 镜像里两者都要有(单独成层, 不作废上面的大缓存)
RUN apt-get -o Acquire::Retries=8 update \
    && apt-get install -y --no-install-recommends docker-cli docker-compose \
    && rm -rf /var/lib/apt/lists/*

# 平台代码与本地化数据集
COPY app.py metrics_dict.json ./
COPY static ./static
COPY engines ./engines
COPY runs ./runs

# HOST=0.0.0.0: 容器内必须绑非回环地址才能从宿主机访问 (app.py 支持 HOST 覆盖)
ENV PORT=8800 HOST=0.0.0.0
EXPOSE 8800

CMD ["python", "app.py"]
