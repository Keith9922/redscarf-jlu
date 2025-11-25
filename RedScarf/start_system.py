#!/usr/bin/env python3
"""
红领巾与敬礼检测系统 - 交互式启动菜单
"""
import sys
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("\n检查环境...")
    
    required = [('cv2', 'opencv-python'), ('numpy', 'numpy'), 
                ('torch', 'torch'), ('ultralytics', 'ultralytics'), 
                ('gradio', 'gradio')]
    
    missing = []
    for module, package in required:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n请安装: pip install {' '.join(missing)}")
        return False
    return True


def check_models():
    """检查模型文件"""
    print("\n检查模型...")
    
    root_dir = Path(__file__).parent
    models = {
        'yolov8n.pt': '人体检测',
        'yolov8n-pose.pt': '姿态检测'
    }
    
    all_exists = True
    for model_file, desc in models.items():
        model_path = root_dir / model_file
        if model_path.exists():
            size = model_path.stat().st_size / 1024 / 1024
            print(f"✅ {desc}: {model_file} ({size:.1f}MB)")
        else:
            print(f"❌ {desc}: {model_file}")
            all_exists = False
    
    return all_exists


def download_missing_models():
    """下载缺失的模型"""
    root_dir = Path(__file__).parent
    pose_model = root_dir / 'yolov8n-pose.pt'
    
    if not pose_model.exists():
        print("\n姿态模型不存在")
        resp = input("是否现在下载? (Y/n): ")
        if resp.lower() != 'n':
            import subprocess
            result = subprocess.run([sys.executable, 'download_pose_model.py'], cwd=root_dir)
            return result.returncode == 0
        print("⚠️  跳过下载，敬礼检测功能将被禁用")
    return True


def launch_app():
    """启动Web应用"""
    print("\n启动Web应用...")
    print("访问: http://localhost:7860")
    print("按 Ctrl+C 停止\n")
    
    try:
        from app import GradioApp
        GradioApp().launch()
    except KeyboardInterrupt:
        print("\n系统已关闭")
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()


def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("🎓 红领巾与敬礼检测系统")
    print("="*60)
    print("\n1. 启动Web界面")
    print("2. 下载姿态模型")
    print("3. 检查系统环境")
    print("0. 退出")
    
    return input("\n请选择 (0-3): ")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎓 红领巾与敬礼检测系统 v4.0")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查模型
    models_ok = check_models()
    
    # 下载缺失的模型
    if not models_ok:
        if not download_missing_models():
            print("\n模型下载失败")
            return
    
    # 交互式菜单
    while True:
        choice = show_menu()
        
        if choice == '1':
            launch_app()
            break
        elif choice == '2':
            download_missing_models()
        elif choice == '3':
            check_environment()
            check_models()
        elif choice == '0':
            print("\n再见！")
            break
        else:
            print("\n无效选项")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
