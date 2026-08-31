#!/bin/bash
# 判分器 uv 预置(taskroot) —— 修 "github 断链 → uvx: command not found → 冤案 0 分"
#
# 背景: TB 2.x 判分脚本(test.sh)普遍先 `curl astral.sh/uv/0.9.5/install.sh | sh` 从
# github release-assets 下载 uv/uvx, 再 `uvx -p 3.13 -w pytest` 时 uv 又要从 github
# 下载托管 CPython。国内经 Clash 到 release-assets.githubusercontent.com 间歇 SSL 断链
# (2026-08-25 schemelike 实锤: 智能体 48 步解出, 判分 0 分)。
#
# 本脚本把判分依赖预置到 runs/tbench/taskroot/, run_bench.py 会自动窄挂载进任务容器:
#   bin/{uv,uvx,env}                  -> /root/.local/bin
#   share/uv/python/cpython-3.13.9-*  -> /root/.local/share/uv/python
# 判分脚本照常跑官方安装器(失败无害), 之后 source env / uvx 用的都是预置件;
# pytest 等包从 PyPI 直连白名单拉(不走代理, 快且稳)。
#
# 注意: 目录名必须是 uv 的命名(cpython-3.13.9-linux-x86_64-gnu, 无日期标签)且带 BUILD
# 标记, uv 才认 —— 所以用 `uv python install` 在 amd64 容器里装, 而不是手工解包。
# TB 判分若升级 uv 版本(TBENCH_UV_VERSION), 重跑本脚本即可换钉版。
#
# 依赖: Docker + 可达代理(自动探测 Clash 7897/7898; 均不可达则直连)。
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TR="$HERE/../../runs/tbench/taskroot"
UV_VER="${TBENCH_UV_VERSION:-0.9.5}"

PROXY=""
for p in 7897 7898; do
  if curl -s --max-time 4 -x "http://127.0.0.1:$p" -o /dev/null https://github.com/; then
    PROXY="http://127.0.0.1:$p"; break
  fi
done
CURL_OPTS=(); [ -n "$PROXY" ] && CURL_OPTS=(-x "$PROXY")
echo "[seed] 代理: ${PROXY:-直连}"

mkdir -p "$TR/bin" "$TR/share/uv/python"
if [ ! -x "$TR/bin/uv" ]; then
  echo "[seed] 下载 uv $UV_VER (linux x86_64)..."
  curl -sL --max-time 300 "${CURL_OPTS[@]}" -o /tmp/uv-seed.tar.gz \
    "https://github.com/astral-sh/uv/releases/download/$UV_VER/uv-x86_64-unknown-linux-gnu.tar.gz"
  (cd /tmp && rm -rf uv-x86_64-unknown-linux-gnu && tar xzf uv-seed.tar.gz)
  cp /tmp/uv-x86_64-unknown-linux-gnu/uv /tmp/uv-x86_64-unknown-linux-gnu/uvx "$TR/bin/"
  chmod +x "$TR/bin/uv" "$TR/bin/uvx"
fi
printf '#!/bin/sh\nexport PATH="/root/.local/bin:$PATH"\n' > "$TR/bin/env"

if [ -z "$(ls -A "$TR/share/uv/python" 2>/dev/null)" ]; then
  # 容器优先复用本地已有的任务镜像(免拉 bookworm), 没有再用 debian:bookworm-slim
  IMG="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -m1 '^alexgshaw/' || echo debian:bookworm-slim)"
  PPORT="${PROXY##*:}"
  echo "[seed] 用 uv 安装托管 CPython 3.13 (镜像: $IMG)..."
  docker run --rm --platform linux/amd64 \
    -v "$TR/bin:/root/.local/bin" -v "$TR/share/uv/python:/tmp/pys" \
    ${PROXY:+-e HTTPS_PROXY=http://host.docker.internal:$PPORT -e HTTP_PROXY=http://host.docker.internal:$PPORT} \
    -e NO_PROXY=localhost,127.0.0.1 -e UV_PYTHON_INSTALL_DIR=/tmp/pys \
    "$IMG" bash -c 'source /root/.local/bin/env && uv python install 3.13'
fi
echo "[seed] 完成: $TR ($(du -sh "$TR" | cut -f1)) · 判分器将自动挂载(关闭: TBENCH_SEED_UV=0)"
