# 🎓 红领巾与敬礼检测系统

基于 YOLOv8 的智能视觉检测系统，用于识别小学生红领巾佩戴情况和敬礼姿态标准性。

## ⚡ 快速开始

```bash
# 1. 安装依赖
cd RedScarf
pip install -r requirements.txt

# 2. 启动应用
python app.py
```

浏览器访问：`http://localhost:7860`

## 🎯 核心功能

### 1. 红领巾检测
- 自动识别是否佩戴红领巾
- 绿色框：已佩戴 | 红色框：未佩戴
- 基于色彩过滤和位置判断

### 2. 敬礼姿态识别
- 17个人体关键点检测（YOLOv8-Pose）
- 100分制评分系统
- 多维度分析：手肘角度、手部位置、整体姿态

### 3. 多模式输入
- 📷 图片上传检测
- 🎥 实时摄像头检测
- 🎬 视频文件处理

## 📁 项目结构

```
redscarf/
└── RedScarf/                     # 核心目录
    ├── app.py                    # Gradio Web应用（主入口）
    ├── Main.py                   # 命令行入口
    ├── start_system.py           # 交互式菜单
    ├── detection_service.py      # 检测服务核心
    ├── config.py                 # 全局配置
    ├── detector/                 # 检测模块
    │   ├── pose_detector.py      # YOLOv8-Pose姿态检测
    │   ├── salute_detector.py    # 敬礼算法
    │   └── utils.py              # 工具函数
    ├── yolov8n.pt                # 人体检测模型
    ├── yolov8n-pose.pt           # 姿态检测模型
    ├── download_pose_model.py    # 模型下载工具
    └── requirements.txt          # Python依赖
```

## 🔧 技术架构

### 核心技术栈

| 技术 | 用途 | 说明 |
|------|------|------|
| **YOLOv8** | 目标检测 | 检测人体和红领巾位置 |
| **YOLOv8-Pose** | 姿态估计 | 17个关键点识别 |
| **OpenCV** | 图像处理 | 色彩过滤、绘图、视频流 |
| **PyTorch** | 深度学习 | 模型推理引擎 |
| **Gradio** | Web界面 | 用户交互界面 |
| **NumPy** | 数值计算 | 向量运算、角度计算 |

### 检测流程

```python
# 1. 红领巾检测流程
图像输入 → YOLOv8检测所有物体 → 色彩过滤(HSV红色范围) 
       → 位置匹配(人体框+红领巾框) → 佩戴判定

# 2. 敬礼检测流程
图像输入 → YOLOv8-Pose检测关键点 → 角度计算(肩-肘-腕)
       → 位置分析(手-头距离) → 评分系统 → 姿态判定
```

### 关键算法说明

**1. 色彩过滤（红领巾识别）**
```python
# HSV色彩空间红色范围
lower_red1 = [0, 50, 50]      # 低色调红色
upper_red1 = [10, 255, 255]
lower_red2 = [170, 50, 50]    # 高色调红色
upper_red2 = [180, 255, 255]

# 红色像素比例 > 15% 才认定为红领巾
red_pixel_ratio > 0.15
```

**2. 位置匹配算法**
```python
# 红领巾必须在人体框的上半部分（脖子到胸部）
valid_area = 人体高度的前50%
# 检查红领巾中心点是否在有效区域内
if (红领巾中心Y坐标 < 人体Y1 + 人体高度*0.5):
    判定为佩戴
```

**3. 敬礼评分系统（100分制）**

| 评分项 | 分值 | 判定标准 |
|--------|------|----------|
| 手肘角度 | 30分 | 60°-120°范围内，最佳90° |
| 手部高度 | 35分 | 手腕在头部附近 |
| 手部位置 | 25分 | 手腕在头部横向范围内 |
| 整体姿态 | 10分 | 手肘抬起高于肩膀 |

**评分等级**：
- 90-100分：✅ 标准敬礼
- 75-89分：敬礼姿态良好
- 60-74分：基本正确
- <60分：不标准

## 🚀 使用方式

### 方式一：Web界面（推荐）

```bash
cd RedScarf
python app.py
```

功能：
- 图片上传检测
- 实时摄像头检测（支持鼓励反馈）
- 详细统计信息展示

### 方式二：交互式菜单

```bash
cd RedScarf
python start_system.py
```

提供：
- 环境检查
- 模型下载
- 应用启动

### 方式三：命令行

```bash
cd RedScarf

# 摄像头实时检测
python Main.py --camera 0

# 检测图片
python Main.py --image test.jpg

# 检测视频并保存
python Main.py --video test.mp4 --output result.mp4
```

## ⚙️ 配置参数

编辑 `RedScarf/config.py`：

```python
# 设备配置
DEVICE = "CPU"  # 或 "GPU"（需要CUDA支持）

# 检测阈值
PERSON_CONF_THRESHOLD = 0.5      # 人体检测置信度
REDSCARF_CONF_THRESHOLD = 0.45    # 红领巾检测置信度
POSE_CONF_THRESHOLD = 0.5         # 姿态检测置信度

# 敬礼判定参数
SALUTE_ANGLE_MIN = 60             # 手肘最小角度
SALUTE_ANGLE_MAX = 120            # 手肘最大角度
SALUTE_HAND_HEAD_RATIO = 0.3     # 手头距离比例

# Web服务配置
GRADIO_SERVER_PORT = 7860         # 访问端口
```

## 📊 模型说明

### YOLOv8n.pt（人体检测）
- **大小**: 6.4MB
- **用途**: 检测图像中的人体位置
- **输出**: 人体边界框 [x1, y1, x2, y2]

### YOLOv8n-pose.pt（姿态检测）
- **大小**: 6.7MB
- **用途**: 检测17个人体关键点
- **关键点**: 鼻、眼、耳、肩、肘、腕、髋、膝、踝
- **输出**: 关键点坐标 [[x, y, confidence], ...]

自动下载：
```bash
cd RedScarf
python download_pose_model.py
```

## 🆘 常见问题

**Q1: 依赖安装失败？**
```bash
# 使用国内镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r RedScarf/requirements.txt
```

**Q2: 摄像头无法打开？**
- 检查摄像头权限（macOS需要在系统设置中授权）
- 尝试不同的摄像头ID：0, 1, 2...
- 确认没有其他程序占用摄像头

**Q3: 检测速度慢？**
- CPU模式下正常，约15-25 FPS
- 如有NVIDIA GPU，设置 `DEVICE = "GPU"`
- 降低输入图像分辨率

**Q4: 红领巾误检或漏检？**
- 调整 `REDSCARF_CONF_THRESHOLD` (0.3-0.6)
- 确保光线充足，红领巾颜色鲜艳
- 避免背景中有大面积红色物体

**Q5: 禁用敬礼检测？**
```bash
# 删除姿态模型文件
rm RedScarf/yolov8n-pose.pt
```

## 📝 更新日志

**v4.0** (2024-11)
- ✨ 新增YOLOv8-Pose姿态检测
- ✨ 敬礼动作智能识别与评分
- ✨ 实时摄像头检测支持
- ✨ 自动鼓励反馈系统
- 🔧 优化检测算法和准确率

## 📄 开源协议

Apache License 2.0 - 详见 `RedScarf/LICENSE`

---

**开发**: 傅雷中学王新语 | **技术支持**: AI辅助开发 | **更新**: 2024年11月
