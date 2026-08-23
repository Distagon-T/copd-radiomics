# -*- coding: utf-8 -*-
"""
prepare_package.py — 准备 Docker 打包内容 (在 Windows 上运行一次)
=================================================================
作用：
  1. 复制 Connectivity-Aware-Airway-Segmentation 代码 + airway 权重
  2. 复制 TotalSegmentator 已下载的 nnUNet 权重 (~/.totalsegmentator)
  3. 复制批量分割脚本 (batch_segment_contrast_ct.py 等) 到镜像内
  4. 生成离线 pip 依赖 (requirements.txt)

用法：
  python prepare_package.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # docker-airway-seg/
PROJ = ROOT.parent
SRC_AIRWAY = PROJ / "Connectivity-Aware-Airway-Segmentaion"
SRC_TOTALSEG = Path.home() / ".totalsegmentator" / "nnunet"
SRC_SCRIPTS = [
    PROJ / "batch_segment_contrast_ct.py",
    PROJ / "batch_segment_largest_slice.py",
]
DST_CODE = ROOT / "code" / "airway_seg"
DST_WEIGHTS = ROOT / "weights"
DST_SCRIPTS = ROOT / "scripts"


def rmtree_safe(p: Path):
    if p.exists():
        shutil.rmtree(p)


def copy_airway_code():
    """复制 airway 分割代码（排除 git/缓存/测试数据），权重单独复制。"""
    print("[1/4] 复制 Connectivity-Aware-Airway-Segmentation 代码 ...")
    rmtree_safe(DST_CODE)
    DST_CODE.mkdir(parents=True, exist_ok=True)

    skip = {"__pycache__", ".git", ".idea", "sample", "sample_out",
            "tests", "examples", "challenge_solution", "figs", "data",
            "_test_airway_model.py"}   # 含 Windows 绝对路径的测试脚本，容器内不运行
    for item in SRC_AIRWAY.iterdir():
        if item.name in skip:
            print(f"      跳过 {item.name}")
            continue
        dst = DST_CODE / item.name
        if item.is_dir():
            shutil.copytree(item, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, dst)
    print("      代码复制完成。")
    print(f"      ⚠️  请手工确认: {DST_CODE / 'configs' / 'airway_config.py'} 中 "
          f"weight_path 已改为相对路径 'checkpoints/airway_model.pth'")


def copy_weights():
    """复制 airway 权重 + TotalSeg nnUNet 权重（完全离线）。"""
    print("[2/4] 复制模型权重 ...")
    rmtree_safe(DST_WEIGHTS)
    DST_WEIGHTS.mkdir(parents=True, exist_ok=True)

    # airway 权重
    aw = SRC_AIRWAY / "checkpoints" / "airway_model.pth"
    if aw.exists():
        dst_aw = DST_CODE / "checkpoints" / "airway_model.pth"
        dst_aw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aw, dst_aw)
        print(f"      airway_model.pth ({aw.stat().st_size/1e6:.1f} MB)")
    else:
        print("      ⚠️ 未找到 airway_model.pth")

    # TotalSeg nnUNet 权重
    if SRC_TOTALSEG.exists():
        shutil.copytree(SRC_TOTALSEG, DST_WEIGHTS / "nnunet",
                        ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"))
        gb = sum(p.stat().st_size for p in (DST_WEIGHTS / "nnunet").rglob("*") if p.is_file()) / 1e9
        print(f"      TotalSegmentator nnUNet 权重 ({gb:.2f} GB)")
    else:
        print(f"      ⚠️ 未找到 TotalSeg 权重目录: {SRC_TOTALSEG}")


def copy_scripts():
    """复制批量分割脚本（作为镜像内可直接运行的入口模板）。"""
    print("[3/4] 复制批量分割脚本 ...")
    DST_SCRIPTS.mkdir(parents=True, exist_ok=True)
    for s in SRC_SCRIPTS:
        if s.exists():
            shutil.copy2(s, DST_SCRIPTS / s.name)
            print(f"      {s.name}")


def gen_requirements():
    """基于当前 conda 环境生成 requirements（锁版本，保证镜像与本地一致）。"""
    print("[4/4] 生成 requirements.txt ...")
    pkgs = [
        "torch==2.6.0",
        "monai==1.4.0",
        "SimpleITK==2.4.1",
        "nibabel==5.3.2",
        "numpy",
        "TotalSegmentator==2.13.0",
    ]
    # TotalSegmentator 会自动拉入其依赖 (nnunetv2 等)，这里只列核心
    req = "\n".join(pkgs) + "\n"
    (ROOT / "requirements.txt").write_text(req, encoding="utf-8")
    print(f"      requirements.txt 已生成 (核心包锁版本)")


def main():
    print("=" * 60)
    print("准备 Docker 打包内容")
    print("=" * 60)
    copy_airway_code()
    copy_weights()
    copy_scripts()
    gen_requirements()
    print("\n✅ 准备完成！打包目录: " + str(ROOT))
    print("   下一步: 把整个 docker-airway-seg 文件夹拷到 Linux 服务器，")
    print("   然后在服务器上执行:  docker build -t airway-seg .  ")


if __name__ == "__main__":
    main()
