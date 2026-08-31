#!/bin/zsh
# ragflow 主镜像多源重试拉取 (daocloud 限流 "请求过于频繁" → 多镜像源轮询 + 退避)
TAG=v0.27.0
for round in 1 2 3 4 5 6 7 8 9 10; do
  for m in docker.m.daocloud.io docker.1ms.run hub.atomgit.com; do
    echo "[round $round] 尝试 $m ..."
    if docker pull -q "$m/infiniflow/ragflow:$TAG" >/dev/null 2>&1; then
      docker tag "$m/infiniflow/ragflow:$TAG" "docker.io/infiniflow/ragflow:$TAG"
      echo "SUCCESS via $m"; exit 0
    fi
  done
  echo "[round $round] 全部失败, 90s 后重试"
  sleep 90
done
echo "ALL FAILED"; exit 1
