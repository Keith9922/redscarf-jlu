#!/usr/bin/env python3
"""
红领巾检测系统启动脚本
自动设置工作目录，然后启动应用
"""
import os
import sys
from pathlib import Path

# 获取脚本所在目录
script_dir = Path(__file__).parent
redscarf_dir = script_dir / 'RedScarf'

print("=" * 70)
print("🚀 红领巾检测系统启动")
print("=" * 70)

# 检查RedScarf目录
if not redscarf_dir.exists():
    print(f"❌ 错误: 找不到RedScarf目录: {redscarf_dir}")
    sys.exit(1)

# 切换到RedScarf目录
os.chdir(redscarf_dir)
print(f"✓ 工作目录: {os.getcwd()}")

# 添加当前目录到Python路径
sys.path.insert(0, str(redscarf_dir))

# 检查关键文件
required_files = [
    'app.py',
    'detection_service.py',
    'config.py',
    'yolov8n.pt',
    'data/models/redscarf.pt'
]

print("\n📋 检查必要文件...")
all_exist = True
for file in required_files:
    path = redscarf_dir / file
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ 缺少必要文件，启动失败")
    sys.exit(1)

print("\n✅ 所有必要文件检查完成")
print("\n" + "=" * 70)
print("启动应用...")
print("=" * 70 + "\n")

# 启动应用
from app import main
main()
