#!/bin/bash
# 红领巾检测系统 - 快速启动脚本

echo "=========================================="
echo "    红领巾检测系统 - 快速启动"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "错误: 未找到Python环境"
        exit 1
    fi
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "使用Python: $PYTHON_CMD"
echo ""

# 显示菜单
echo "请选择启动模式:"
echo "  1) Web界面 (推荐)"
echo "  2) 摄像头实时检测"
echo "  3) 检测图片"
echo "  4) 检测视频"
echo "  5) 安装依赖"
echo "  0) 退出"
echo ""

read -p "请输入选项 [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "正在启动Web界面..."
        $PYTHON_CMD app.py
        ;;
    2)
        echo ""
        read -p "请输入摄像头ID [默认0]: " camera_id
        camera_id=${camera_id:-0}
        echo "正在启动摄像头检测 (Camera ID: $camera_id)..."
        $PYTHON_CMD Main.py --camera $camera_id
        ;;
    3)
        echo ""
        read -p "请输入图片路径: " image_path
        if [ -z "$image_path" ]; then
            echo "错误: 未指定图片路径"
            exit 1
        fi
        echo "正在检测图片..."
        $PYTHON_CMD Main.py --image "$image_path"
        ;;
    4)
        echo ""
        read -p "请输入视频路径: " video_path
        if [ -z "$video_path" ]; then
            echo "错误: 未指定视频路径"
            exit 1
        fi
        read -p "是否保存结果? (y/n) [默认n]: " save_result
        if [ "$save_result" = "y" ] || [ "$save_result" = "Y" ]; then
            read -p "请输入输出路径 [默认output.mp4]: " output_path
            output_path=${output_path:-output.mp4}
            echo "正在检测视频并保存..."
            $PYTHON_CMD Main.py --video "$video_path" --output "$output_path"
        else
            echo "正在检测视频..."
            $PYTHON_CMD Main.py --video "$video_path"
        fi
        ;;
    5)
        echo ""
        echo "正在安装依赖..."
        $PYTHON_CMD -m pip install -r requirements.txt
        echo ""
        echo "依赖安装完成!"
        ;;
    0)
        echo "退出程序"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
