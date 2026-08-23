#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线打包工具（在开发机上运行，生成 Docker 构建所需的全部文件）
==============================================================
1) 收集 TotalSegmentator 权重 -> ./weights/nnunet
   （开发机 .totalsegmentator 缓存目录 -> 容器内 /app/weights/nnunet，
     配合 TOTALSEG_HOME_DIR=/app/weights 使用）
2) （可选）生成离线 wheelhouse：把 pip 依赖下载成 wheel，
   供完全无网的服务器构建时用 --find-links 安装。

用法:
  python prepare_package.py                 # 只收集权重
  python prepare_package.py --wheels        # 同时生成 wheelhouse
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 开发机 TotalSegmentator 权重缓存（Linux: ~/.totalsegmentator；Windows 可能不同）
CANDIDATE_WEIGHTS = [
    os.path.expanduser("~/.totalsegmentator"),
    os.path.expanduser("~/.cache/totalsegmentator"),
]


def find_weights():
    for d in CANDIDATE_WEIGHTS:
        if os.path.isdir(d):
            return d
    # 也检查 TOTALSEG_HOME_DIR 环境变量
    env = os.environ.get("TOTALSEG_HOME_DIR", "")
    if env and os.path.isdir(env):
        return env
    return None


def collect_weights():
    dst = os.path.join(HERE, "weights", "nnunet")
    src = find_weights()
    if src is None:
        print("[warn] 未找到 TotalSegmentator 权重缓存，请先运行一次 TotalSegmentator，")
        print("       或手动把权重放到 weights/nnunet/")
        return False
    os.makedirs(dst, exist_ok=True)
    # 复制 nnunet 相关子目录（results/ 等）
    copied = 0
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        copied += 1
    print(f"[ok] 权重已收集: {src} -> {dst} ({copied} 项)")
    return True


def build_wheelhouse():
    req = os.path.join(HERE, "requirements.txt")
    whl = os.path.join(HERE, "wheelhouse")
    os.makedirs(whl, exist_ok=True)
    print(f"下载依赖 wheel 到 {whl} ...")
    # 目标平台: linux x86_64 / python3.11 (ubuntu 22.04, manylinux)
    cmd = [sys.executable, "-m", "pip", "download",
           "--dest", whl, "-r", req,
           "--platform", "manylinux2014_x86_64",
           "--only-binary=:all:",
           "--python-version", "3.11"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)
    print(f"[done] wheelhouse: {whl}")


def main():
    p = argparse.ArgumentParser(description="离线打包工具")
    p.add_argument("--wheels", action="store_true", help="同时生成离线 wheelhouse")
    args = p.parse_args()

    print("== 收集 TotalSegmentator 权重 ==")
    collect_weights()
    if args.wheels:
        print("\n== 生成离线 wheelhouse ==")
        build_wheelhouse()
    print("\n完成。现在可以构建 Docker:")
    print("  docker build -t radiomics-seg:latest .")


if __name__ == "__main__":
    main()
