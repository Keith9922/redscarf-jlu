# 红领巾检测系统

基于YOLO的红领巾佩戴检测系统，支持实时图像和摄像头检测。

## 快速启动

```bash
# 安装依赖
pip install -r RedScarf/requirements.txt

# 启动应用
python run_app.py
```

## 项目结构

- `RedScarf/` - 主项目目录
  - `app.py` - Gradio Web界面
  - `detection_service.py` - 检测服务核心
  - `start.sh` / `start.bat` - 启动脚本
  - `data/models/` - 模型文件
  - `detector/` - 检测器模块
  - `src/` - 源代码

## 功能特性

- 红领巾佩戴检测
- 敬礼姿态识别
- 实时摄像头检测
- 图像批量处理

## 技术栈

- YOLOv8 目标检测
- YOLOv8-Pose 姿态估计
- Gradio Web界面
- OpenCV 图像处理
