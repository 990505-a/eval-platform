# -*- coding: utf-8 -*-
"""代理端口桥: 0.0.0.0:7898 → 127.0.0.1:7897 (宿主机 Clash)。

用途: Docker 容器经 host.docker.internal:7898 使用宿主机 Clash 代理
(容器到 host.docker.internal 走的是虚拟网卡, 够不着只绑 127.0.0.1 的服务,
本桥监听全部接口转发到回环, 解决这一层)。
启动: nohup python3 scripts/proxy-bridge.py >> scripts/proxy-bridge.log 2>&1 &
"""
import socket
import threading

LISTEN = ("0.0.0.0", 7898)
TARGET = ("127.0.0.1", 7897)


def _pipe(a: socket.socket, b: socket.socket):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


def main():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(LISTEN)
    srv.listen(128)
    print(f"proxy-bridge: {LISTEN} → {TARGET}", flush=True)
    while True:
        client, _ = srv.accept()
        try:
            upstream = socket.socket()
            upstream.connect(TARGET)
        except OSError:
            client.close()
            continue
        threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
        threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()


if __name__ == "__main__":
    main()
