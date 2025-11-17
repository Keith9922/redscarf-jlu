#!/usr/bin/env python3
"""
下载YOLOv8-Pose模型
自动从Ultralytics下载预训练的YOLOv8n-Pose模型
"""
from pathlib import Path
from ultralytics import YOLO


def download_pose_model():
    """下载YOLOv8-Pose模型"""
    print("="*60)
    print("下载YOLOv8-Pose模型")
    print("="*60)
    
    # 获取项目根目录
    root_dir = Path(__file__).parent
    model_path = root_dir / 'yolov8n-pose.pt'
    
    # 检查模型是否已存在
    if model_path.exists():
        print(f"\n[INFO] 模型已存在: {model_path}")
        print(f"[INFO] 模型大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        response = input("\n是否重新下载? (y/N): ")
        if response.lower() != 'y':
            print("[INFO] 跳过下载")
            return True
    
    print("\n[INFO] 开始下载YOLOv8n-Pose模型...")
    print("[INFO] 这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 使用YOLO类自动下载模型
        # 第一次调用会自动从Ultralytics下载
        model = YOLO('yolov8n-pose.pt')
        
        # 保存到项目目录
        import shutil
        
        # 查找下载的模型文件
        # Ultralytics通常会下载到 ~/.cache/ultralytics 或当前目录
        downloaded_paths = [
            Path.home() / '.cache' / 'ultralytics' / 'yolov8n-pose.pt',
            Path('yolov8n-pose.pt')
        ]
        
        source_path = None
        for path in downloaded_paths:
            if path.exists():
                source_path = path
                break
        
        if source_path and source_path != model_path:
            shutil.copy(source_path, model_path)
            print(f"\n[INFO] 模型已复制到: {model_path}")
        elif model_path.exists():
            print(f"\n[INFO] 模型已下载: {model_path}")
        else:
            print("\n[WARNING] 未找到下载的模型文件，但模型已加载到内存")
            print("[INFO] 可以正常使用，但建议手动下载模型文件")
        
        # 验证模型
        print("\n[INFO] 验证模型...")
        model = YOLO(str(model_path) if model_path.exists() else 'yolov8n-pose.pt')
        print("[INFO] 模型验证成功！")
        
        # 显示模型信息
        print("\n模型信息:")
        print(f"  - 模型名称: YOLOv8n-Pose")
        print(f"  - 用途: 人体姿态估计 (17个关键点)")
        print(f"  - 模型路径: {model_path if model_path.exists() else '缓存中'}")
        if model_path.exists():
            print(f"  - 文件大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        print("\n✅ 模型下载完成！")
        return True
    
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n解决方案:")
        print("1. 检查网络连接")
        print("2. 手动下载模型:")
        print("   - 访问: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")
        print(f"   - 保存到: {root_dir / 'yolov8n-pose.pt'}")
        return False


def main():
    """主函数"""
    print("\nYOLOv8-Pose 模型下载工具\n")
    
    try:
        success = download_pose_model()
        
        if success:
            print("\n" + "="*60)
            print("下载完成！现在可以运行敬礼检测功能了")
            print("="*60)
            print("\n使用方法:")
            print("1. 测试敬礼检测: python test_salute.py")
            print("2. 启动Web界面: python app.py")
        else:
            print("\n" + "="*60)
            print("下载失败，请查看上述解决方案")
            print("="*60)
    
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断下载")
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
