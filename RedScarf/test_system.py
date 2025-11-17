#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统测试脚本 - 验证环境和基本功能
"""

import sys
import importlib

def test_imports():
    """测试所有必需的库是否已正确安装"""
    print("=" * 60)
    print("测试依赖包安装状态")
    print("=" * 60)
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'openvino': 'openvino-dev',
        'ultralytics': 'ultralytics',
        'gradio': 'gradio',
        'PIL': 'Pillow',
    }
    
    all_ok = True
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name:20s} - 已安装")
        except ImportError:
            print(f"❌ {package_name:20s} - 未安装")
            all_ok = False
    
    print()
    return all_ok


def test_project_structure():
    """测试项目文件结构"""
    print("=" * 60)
    print("测试项目文件结构")
    print("=" * 60)
    
    from pathlib import Path
    
    required_files = [
        'config.py',
        'detection_service.py',
        'app.py',
        'Main.py',
        'Log.py',
        'requirements.txt',
        'detector/persondetector.py',
        'detector/redscarfdetector.py',
        'detector/utils.py',
        'models/yolov8n_openvino_model',
        'models/redscarf_openvino_model',
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {str(path):40s} - 存在")
        else:
            print(f"❌ {str(path):40s} - 缺失")
            all_ok = False
    
    print()
    return all_ok


def test_config():
    """测试配置文件"""
    print("=" * 60)
    print("测试配置文件")
    print("=" * 60)
    
    try:
        import config
        
        # 检查关键配置项
        attrs = [
            'PERSON_MODEL_PATH',
            'REDSCARF_MODEL_PATH',
            'DEVICE',
            'PERSON_CONF_THRESHOLD',
            'REDSCARF_CONF_THRESHOLD',
            'GRADIO_SERVER_PORT',
        ]
        
        all_ok = True
        for attr in attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                print(f"✅ {attr:30s} = {value}")
            else:
                print(f"❌ {attr:30s} - 未定义")
                all_ok = False
        
        print()
        return all_ok
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print()
        return False


def test_models():
    """测试模型文件"""
    print("=" * 60)
    print("测试模型文件")
    print("=" * 60)
    
    try:
        from pathlib import Path
        import config
        
        # 检查人体检测模型
        if config.PERSON_MODEL_PATH.exists():
            print(f"✅ 人体检测模型: {config.PERSON_MODEL_PATH}")
        else:
            print(f"❌ 人体检测模型缺失: {config.PERSON_MODEL_PATH}")
            return False
        
        # 检查红领巾检测模型
        if config.REDSCARF_MODEL_PATH.exists():
            print(f"✅ 红领巾检测模型: {config.REDSCARF_MODEL_PATH}")
        else:
            print(f"❌ 红领巾检测模型缺失: {config.REDSCARF_MODEL_PATH}")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        print()
        return False


def test_detection_service():
    """测试检测服务初始化"""
    print("=" * 60)
    print("测试检测服务")
    print("=" * 60)
    
    try:
        from detection_service import RedScarfDetectionService
        
        print("正在初始化检测服务...")
        detector = RedScarfDetectionService()
        print("✅ 检测服务初始化成功")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 检测服务初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("红领巾检测系统 - 环境测试")
    print("=" * 60)
    print()
    
    # 运行所有测试
    tests = [
        ("依赖包检查", test_imports),
        ("文件结构检查", test_project_structure),
        ("配置文件检查", test_config),
        ("模型文件检查", test_models),
        ("检测服务检查", test_detection_service),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            results[test_name] = False
    
    # 输出总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    print()
    print(f"总计: {passed}/{total} 通过")
    print()
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        print()
        print("您可以通过以下方式启动系统:")
        print("  1. Web界面:    python app.py")
        print("  2. 摄像头检测:  python Main.py")
        print("  3. 快速启动:   ./start.sh (Linux/macOS) 或 start.bat (Windows)")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        print()
        print("常见解决方案:")
        print("  1. 安装缺失的依赖: pip install -r requirements.txt")
        print("  2. 检查模型文件是否完整")
        print("  3. 查看 README_USAGE.md 获取更多帮助")
        return 1


if __name__ == "__main__":
    sys.exit(main())
