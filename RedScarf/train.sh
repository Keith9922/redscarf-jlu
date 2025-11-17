#!/bin/bash
# 红领巾检测模型训练 - Mac M系列芯片优化版

echo "=========================================="
echo "🎓 红领巾检测模型训练脚本"
echo "=========================================="
echo ""

# 激活conda环境
echo "📦 激活conda环境: red"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate red

# 检查环境
echo ""
echo "🔍 检查训练环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'MPS可用: {torch.backends.mps.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO已安装')"

echo ""
echo "=========================================="
echo "🚀 开始训练"
echo "=========================================="
echo ""

# 训练参数
EPOCHS=${1:-100}        # 默认100轮
BATCH=${2:-16}          # 默认batch=16
DEVICE=${3:-mps}        # 默认使用MPS加速

echo "训练参数:"
echo "  - 训练轮数: $EPOCHS"
echo "  - 批次大小: $BATCH"
echo "  - 训练设备: $DEVICE"
echo ""

# 执行训练
python train_redscarf.py \
    --data data/datasets/data.yaml \
    --model yolov8n.pt \
    --epochs $EPOCHS \
    --batch $BATCH \
    --device $DEVICE \
    --name redscarf_training_$(date +%Y%m%d_%H%M%S) \
    --patience 20

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
