#!/usr/bin/env python3
"""
快速启动摄像头检测脚本
直接运行即可启动Web界面并支持摄像头实时检测
"""
import os
import sys
from pathlib import Path

# 获取脚本所在目录
script_dir = Path(__file__).parent
redscarf_dir = script_dir / 'RedScarf'

print("=" * 70)
print("🚀 红领巾检测系统 - 摄像头实时检测模式")
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
print("启动Web界面...")
print("=" * 70 + "\n")

# 启动应用
try:
    from app import GradioApp
    
    app = GradioApp()
    print("\n💡 使用说明:")
    print("  1. Web界面启动后，点击打开的URL")
    print("  2. 进入'🎥 摄像头实时检测'标签页")
    print("  3. 点击'▶️ 启动摄像头'开始实时检测")
    print("  4. 当检测到正确佩戴红领巾且敬礼时，会自动显示鼓励信息")
    print("  5. 点击'⏹️ 停止摄像头'结束检测")
    print("\n按 Ctrl+C 关闭应用\n")
    
    app.launch()
except KeyboardInterrupt:
    print("\n[INFO] 系统已关闭")
except Exception as e:
    print(f"\n[ERROR] 系统错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
