# -*- coding: utf-8 -*-
"""打包 eval-platform -> ../eval-platform.zip (排除环境/缓存/日志/真实密钥文件)"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r"D:\job\eval-platform")
OUT = Path(r"D:\job\eval-platform.zip")

EXCLUDE_DIRS = {"node_modules", "__pycache__", ".venv-opencompass", ".cache",
                "icl_inference_output", "tmp", "work", ".git", ".zcode", ".idea", ".vscode"}
EXCLUDE_FILES = {".env", "module_config.json", "_make_zip.py",
                 "mc.json", "last_eval.json", "state.json",
                 "models.json",  # 模型登记(含真实 API key)
                 "result.json", "dyn_redteam.yaml", ".archived"}

def keep(rel: str, name: str) -> bool:
    if name in EXCLUDE_FILES:
        return False
    if name.endswith((".log", ".pid", ".tmp", ".pyc")):
        return False
    # runs/ 只保留成绩与归档(llm 结果 + redteam archive), 其余为过程产物
    if rel.startswith("runs\\") and not (rel.startswith("runs\\llm") or rel.startswith("runs\\redteam\\archive")):
        return False
    return True

n_files, n_dirs, total = 0, 0, 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    # os.walk 自顶向下, 在 dirs 里原地剪掉排除目录(含其全部子内容)
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_dir = Path(dirpath).relative_to(ROOT)
        for d in dirs:
            rel = str(rel_dir / d) if str(rel_dir) != "." else d
            z.writestr(zipfile.ZipInfo("eval-platform/" + rel.replace("\\", "/") + "/"), "")
            n_dirs += 1
        for f in sorted(files):
            rel = str(rel_dir / f) if str(rel_dir) != "." else f
            if not keep(rel, f):
                continue
            p = Path(dirpath) / f
            z.write(p, "eval-platform/" + rel.replace("\\", "/"))
            n_files += 1
            total += p.stat().st_size

print(f"files={n_files} dirs={n_dirs} raw={total/1024/1024:.1f}MB zip={OUT.stat().st_size/1024/1024:.1f}MB")
