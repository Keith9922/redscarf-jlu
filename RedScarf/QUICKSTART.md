# 快速开始指南

## 🚀 一键启动

```bash
cd RedScarf
python start_system.py
```

该脚本会自动：
1. ✅ 检查Python依赖包
2. ✅ 检查模型文件
3. ✅ 下载缺失的模型
4. ✅ 启动Web界面

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install opencv-python numpy torch ultralytics gradio
```

## 🎯 主要功能

### 1. 红领巾检测
- ✅ 自动识别人体
- ✅ 检测红领巾位置
- ✅ 判断佩戴状态

### 2. 敬礼姿态识别（新功能）
- ✅ 人体关键点检测
- ✅ 敬礼动作识别
- ✅ 姿态标准评分
- ✅ 详细反馈信息

## 📥 模型下载

### 自动下载（推荐）
```bash
python download_pose_model.py
```

### 手动下载
如果自动下载失败，请访问：
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

下载后放置到 `RedScarf/yolov8n-pose.pt`

## 🧪 测试功能

### 测试敬礼检测
```bash
python test_salute.py
```

### 测试完整系统
```bash
python test_system.py
```

## 🌐 Web界面

启动后访问：http://localhost:7860

### 功能特性
- 📤 上传图片
- 🔍 一键检测
- 📊 结果可视化
- 📈 详细统计信息

### 检测结果说明
- 🟢 绿色框 = 已佩戴红领巾
- 🔴 红色框 = 未佩戴红领巾
- 🔵 青色框 = 红领巾位置
- 🟣 紫色框 = 标准敬礼姿态
- 🟡 黄色骨架 = 人体关键点

## ⚙️ 配置参数

编辑 `config.py` 调整参数：

```python
# 姿态检测
POSE_CONF_THRESHOLD = 0.5        # 置信度阈值

# 敬礼检测
SALUTE_ANGLE_MIN = 60            # 最小手肘角度
SALUTE_ANGLE_MAX = 120           # 最大手肘角度
SALUTE_STRICT_MODE = False       # 严格模式
```

## 📖 详细文档

查看完整文档：[敬礼检测功能使用指南](SALUTE_DETECTION_GUIDE.md)

## 🆘 常见问题

### Q: 依赖安装失败？
A: 使用国内镜像源
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### Q: 模型下载慢？
A: 使用手动下载方式，见上方说明

### Q: 敬礼检测不准？
A: 
1. 确保图像清晰
2. 人物姿态完整
3. 调整配置参数

### Q: 如何禁用敬礼检测？
A: 删除 `yolov8n-pose.pt` 模型文件，系统会自动禁用该功能

## 📞 技术支持

遇到问题请查看：
- 详细文档：`SALUTE_DETECTION_GUIDE.md`
- 原项目README：`readme.md`
- 提交Issue获取帮助

## 🔄 版本信息

当前版本: **v3.0**

新增功能：
- ✨ YOLOv8-Pose姿态检测
- ✨ 敬礼动作识别
- ✨ 姿态评分系统
- ✨ Web界面增强
