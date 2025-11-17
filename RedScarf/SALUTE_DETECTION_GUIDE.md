# 敬礼检测功能使用指南

## 功能简介

本项目在原有红领巾检测的基础上，新增了**敬礼姿态识别**功能，使用YOLOv8-Pose模型进行人体关键点检测，并基于关键点算法判断小学生敬礼动作是否标准。

## 技术架构

### 核心技术
- **YOLOv8**: 人体检测
- **YOLOv8-Pose**: 人体姿态估计（17个关键点）
- **自定义算法**: 敬礼姿态评分算法
- **Gradio**: Web前端界面

### 检测流程
1. **人体检测**: 检测图像中的所有人体
2. **红领巾检测**: 识别红领巾佩戴情况
3. **姿态检测**: 使用YOLOv8-Pose检测17个人体关键点
4. **敬礼判断**: 基于关键点计算手肘角度、手部位置等特征
5. **结果输出**: 综合评分和详细反馈

## 快速开始

### 1. 下载YOLOv8-Pose模型

```bash
cd RedScarf
python download_pose_model.py
```

该脚本会自动下载`yolov8n-pose.pt`模型文件（约6MB）。

**手动下载**（如果自动下载失败）:
```bash
# 下载链接
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

# 或使用curl
curl -L -o yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

### 2. 测试敬礼检测功能

```bash
python test_salute.py
```

此脚本会：
- 加载YOLOv8-Pose模型
- 测试姿态检测功能
- 识别敬礼动作
- 显示检测结果和评分

### 3. 启动Web界面

```bash
python app.py
```

访问 `http://localhost:7860` 使用完整功能。

## 敬礼判断标准

### 评分维度（总分100分）

| 维度 | 分值 | 标准 |
|------|------|------|
| 手肘角度 | 30分 | 60°-120° 范围内，最佳角度90° |
| 手部高度 | 35分 | 手部应在头部附近 |
| 手部位置 | 25分 | 手部应在头部侧方 |
| 整体姿态 | 10分 | 手肘抬起高于肩膀 |

### 评分等级

- **90-100分**: ✅ 标准敬礼（紫色边框）
- **75-89分**: 敬礼姿态良好
- **60-74分**: 敬礼姿态基本正确
- **<60分**: 未敬礼或姿态不标准

### 关键点检测

YOLOv8-Pose检测17个人体关键点：

```
0: 鼻子
1-2: 左眼、右眼
3-4: 左耳、右耳
5-6: 左肩、右肩
7-8: 左肘、右肘
9-10: 左手腕、右手腕
11-12: 左髋、右髋
13-14: 左膝、右膝
15-16: 左踝、右踝
```

敬礼检测主要使用：**肩膀、手肘、手腕、头部（鼻子）** 关键点。

## 配置参数

在 `config.py` 中可调整敬礼检测参数：

```python
# 姿态检测参数
POSE_CONF_THRESHOLD = 0.5        # 姿态检测置信度阈值

# 敬礼检测参数
SALUTE_ANGLE_MIN = 60            # 敬礼手肘最小角度
SALUTE_ANGLE_MAX = 120           # 敬礼手肘最大角度
SALUTE_HAND_HEAD_RATIO = 0.3    # 手部到头部距离比例
SALUTE_STRICT_MODE = False       # 是否使用严格模式
```

## 代码结构

### 新增文件

```
RedScarf/
├── detector/
│   ├── pose_detector.py       # YOLOv8-Pose姿态检测器
│   └── salute_detector.py     # 敬礼检测算法
├── download_pose_model.py     # 模型下载脚本
├── test_salute.py             # 敬礼检测测试脚本
└── yolov8n-pose.pt           # YOLOv8-Pose模型（需下载）
```

### 更新文件

- `config.py`: 添加姿态检测配置
- `detection_service.py`: 集成姿态检测和敬礼识别
- `app.py`: 更新Web界面显示敬礼结果

## API 使用示例

### Python 代码示例

```python
from detector.pose_detector import PoseDetector
from detector.salute_detector import SaluteDetector
import cv2

# 初始化检测器
pose_detector = PoseDetector(
    model_path='yolov8n-pose.pt',
    device='CPU',
    conf_threshold=0.5
)

salute_detector = SaluteDetector(
    angle_threshold=(60, 120),
    strict_mode=False
)

# 读取图像
image = cv2.imread('test.jpg')

# 姿态检测
pose_detections = pose_detector.detect(image)

# 敬礼检测
for detection in pose_detections:
    keypoints = detection['keypoints']
    result = salute_detector.detect_salute(keypoints)
    
    print(f"是否敬礼: {result['is_saluting']}")
    print(f"姿态得分: {result['score']:.1f}/100")
    print(f"详细信息: {result['details']}")
```

### 集成检测服务

```python
from detection_service import RedScarfDetectionService

# 初始化服务（自动启用姿态检测）
detector = RedScarfDetectionService(
    device='CPU',
    enable_pose=True
)

# 检测图像
result_image, info = detector.detect_image(image)

# 输出结果
print(f"检测到 {info['total_persons']} 人")
print(f"正在敬礼: {info['saluting']} 人")

# 查看敬礼详情
for salute_result in info['salute_results']:
    if salute_result['is_saluting']:
        print(f"得分: {salute_result['score']:.1f}")
        print(f"评价: {salute_result['details']['posture']}")
```

## 常见问题

### Q1: 姿态检测不准确怎么办？

**解决方案**:
1. 确保图像清晰，光线充足
2. 人物姿态完整可见
3. 调整 `POSE_CONF_THRESHOLD` 参数
4. 使用更高分辨率的图像

### Q2: 敬礼判断过于严格/宽松？

**解决方案**:
- 严格模式: 设置 `SALUTE_STRICT_MODE = True` （最低70分）
- 宽松模式: 设置 `SALUTE_STRICT_MODE = False` （最低60分）
- 调整角度范围: `SALUTE_ANGLE_MIN` 和 `SALUTE_ANGLE_MAX`

### Q3: 模型下载失败？

**解决方案**:
1. 检查网络连接
2. 使用手动下载（见快速开始）
3. 使用国内镜像或VPN

### Q4: 如何禁用敬礼检测？

**方法1 - 启动参数**:
```python
detector = RedScarfDetectionService(enable_pose=False)
```

**方法2 - 删除模型文件**:
```bash
rm yolov8n-pose.pt
```

系统会自动检测并禁用姿态功能。

## 性能优化

### 推理速度

| 硬件配置 | 推理速度 | 说明 |
|---------|---------|------|
| CPU (8核) | 5-10 FPS | 基础配置 |
| GPU (NVIDIA) | 20-30 FPS | 推荐配置 |

### GPU加速

```python
# 使用GPU
detector = RedScarfDetectionService(device='GPU')
```

需要安装CUDA和对应的PyTorch GPU版本。

## 训练自定义模型

如需训练更精准的敬礼检测模型，可以：

1. 收集敬礼姿态数据集
2. 标注关键点
3. 使用YOLOv8-Pose进行微调

详见 Ultralytics 官方文档: https://docs.ultralytics.com/tasks/pose/

## 许可证

本项目遵循原项目许可证。

## 更新日志

### v3.0 (2024-11-15)
- ✨ 新增YOLOv8-Pose姿态检测
- ✨ 新增敬礼动作识别算法
- ✨ Web界面显示敬礼检测结果
- 📝 完善文档和测试脚本

### v2.0
- 红领巾检测基础功能

## 联系方式

如有问题或建议，欢迎提Issue。
