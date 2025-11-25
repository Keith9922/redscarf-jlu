#!/usr/bin/env python3
"""
下载YOLOv8-Pose模型
"""
from pathlib import Path
from ultralytics import YOLO


def download_pose_model():
    """下载YOLOv8-Pose模型"""
    root_dir = Path(__file__).parent
    model_path = root_dir / 'yolov8n-pose.pt'
    
    # 检查模型是否已存在
    if model_path.exists():
        print(f"✅ 模型已存在: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    
    print("\n正在下载YOLOv8-Pose模型...")
    
    try:
        # 使用YOLO类自动下载
        model = YOLO('yolov8n-pose.pt')
        print("✅ 模型下载成功！")
        return True
    
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n手动下载地址:")
        print("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print("📦 YOLOv8-Pose 模型下载")
    print("="*60)
    
    if download_pose_model():
        print("\n✅ 完成！现在可以运行敬礼检测功能")
    else:
        print("\n❌ 请手动下载模型")


if __name__ == "__main__":
    main()
