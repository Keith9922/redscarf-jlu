#!/usr/bin/env python3
"""
红领巾与敬礼检测系统 - 完整功能启动脚本
包含模型检查、下载和系统启动
"""
import sys
from pathlib import Path


def check_environment():
    """检查运行环境"""
    print("="*60)
    print("环境检查")
    print("="*60)
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('gradio', 'gradio')
    ]
    
    missing_packages = []
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n缺少依赖包，请运行:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_models():
    """检查模型文件"""
    print("\n" + "="*60)
    print("模型文件检查")
    print("="*60)
    
    root_dir = Path(__file__).parent
    
    models = {
        'yolov8n.pt': '人体检测模型',
        'yolov8n-pose.pt': '姿态检测模型（敬礼功能）'
    }
    
    all_exists = True
    
    for model_file, description in models.items():
        model_path = root_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"✅ {description}: {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {description}: {model_file} 不存在")
            all_exists = False
    
    return all_exists


def download_missing_models():
    """下载缺失的模型"""
    root_dir = Path(__file__).parent
    pose_model_path = root_dir / 'yolov8n-pose.pt'
    
    if not pose_model_path.exists():
        print("\n" + "="*60)
        print("下载姿态检测模型")
        print("="*60)
        
        response = input("\n姿态检测模型不存在，是否现在下载? (Y/n): ")
        if response.lower() != 'n':
            import subprocess
            result = subprocess.run(
                [sys.executable, 'download_pose_model.py'],
                cwd=root_dir
            )
            return result.returncode == 0
        else:
            print("\n⚠️  跳过下载，敬礼检测功能将被禁用")
            return True
    
    return True


def run_tests():
    """运行测试"""
    print("\n" + "="*60)
    print("功能测试")
    print("="*60)
    
    response = input("\n是否运行敬礼检测测试? (y/N): ")
    if response.lower() == 'y':
        import subprocess
        root_dir = Path(__file__).parent
        subprocess.run([sys.executable, 'test_salute.py'], cwd=root_dir)


def launch_app():
    """启动Web应用"""
    print("\n" + "="*60)
    print("启动Web应用")
    print("="*60)
    
    print("\n正在启动红领巾与敬礼检测系统...")
    print("Web界面将在浏览器中打开")
    print("访问地址: http://localhost:7860")
    print("\n按 Ctrl+C 停止服务\n")
    
    try:
        from app import GradioApp
        app = GradioApp()
        app.launch()
    except KeyboardInterrupt:
        print("\n\n系统已关闭")
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()


def show_menu():
    """显示菜单"""
    print("\n" + "="*60)
    print("红领巾与敬礼检测系统")
    print("="*60)
    print("\n请选择操作:")
    print("1. 启动Web界面")
    print("2. 测试敬礼检测功能")
    print("3. 下载姿态检测模型")
    print("4. 检查系统环境")
    print("0. 退出")
    
    choice = input("\n请输入选项 (0-4): ")
    return choice


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎓 红领巾与敬礼检测系统 v3.0")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查模型
    models_ok = check_models()
    
    # 下载缺失的模型
    if not models_ok:
        if not download_missing_models():
            print("\n模型下载失败，请手动下载")
            return
    
    # 交互式菜单
    while True:
        choice = show_menu()
        
        if choice == '1':
            launch_app()
            break
        elif choice == '2':
            run_tests()
        elif choice == '3':
            download_missing_models()
        elif choice == '4':
            check_environment()
            check_models()
        elif choice == '0':
            print("\n再见！")
            break
        else:
            print("\n无效选项，请重新选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
