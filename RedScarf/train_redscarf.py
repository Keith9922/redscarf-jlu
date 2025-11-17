#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
红领巾检测模型训练脚本
支持Mac M系列芯片的MPS加速
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import argparse


def check_device():
    """检查可用的训练设备"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ 检测到CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"✅ 检测到Apple Silicon芯片，使用MPS加速")
    else:
        device = 'cpu'
        print(f"⚠️  使用CPU训练（速度较慢）")
    
    return device


def train_model(
    data_yaml='data/datasets/data.yaml',
    base_model='yolov8n.pt',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='mps',
    project='runs/detect',
    name='redscarf_training',
    patience=20,
    save_period=10
):
    """
    训练红领巾检测模型
    
    Args:
        data_yaml: 数据集配置文件路径
        base_model: 基础模型（预训练权重）
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像尺寸
        device: 训练设备 (cpu/mps/cuda)
        project: 项目保存路径
        name: 训练任务名称
        patience: 早停耐心值
        save_period: 模型保存周期
    """
    
    print("=" * 80)
    print("🎓 红领巾检测模型训练")
    print("=" * 80)
    print()
    
    # 检查数据集配置文件
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    print(f"📁 数据集配置: {data_yaml}")
    print(f"🤖 基础模型: {base_model}")
    print(f"📊 训练参数:")
    print(f"   - 训练轮数: {epochs}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 图像尺寸: {img_size}")
    print(f"   - 训练设备: {device}")
    print(f"   - 早停耐心: {patience}")
    print(f"   - 保存周期: 每{save_period}轮")
    print()
    
    # 加载基础模型
    print(f"[INFO] 正在加载基础模型...")
    model = YOLO(base_model)
    
    # 开始训练
    print(f"[INFO] 开始训练...")
    print("=" * 80)
    print()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        exist_ok=False,
        # 数据增强参数
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # 学习率参数
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # 其他参数
        workers=8,
        cache=False,
        amp=True,  # 自动混合精度
    )
    
    print()
    print("=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    print()
    print(f"📊 训练结果保存在: {project}/{name}")
    print(f"🎯 最佳模型: {project}/{name}/weights/best.pt")
    print(f"📈 最新模型: {project}/{name}/weights/last.pt")
    print()
    print("💡 下一步:")
    print(f"   1. 查看训练结果: open {project}/{name}/results.png")
    print(f"   2. 验证模型: python -c \"from ultralytics import YOLO; YOLO('{project}/{name}/weights/best.pt').val()\"")
    print(f"   3. 测试检测: python Main.py -i test_image.jpg")
    print()
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='红领巾检测模型训练')
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/datasets/data.yaml',
        help='数据集配置文件路径'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='基础模型路径（默认: yolov8n.pt）'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数（默认: 100）'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='批次大小（默认: 16）'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='图像尺寸（默认: 640）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='训练设备 (auto/cpu/mps/cuda)，默认auto自动检测'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='redscarf_training',
        help='训练任务名称'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='早停耐心值（默认: 20）'
    )
    
    args = parser.parse_args()
    
    # 自动检测设备
    if args.device == 'auto':
        device = check_device()
    else:
        device = args.device
    
    # 开始训练
    try:
        train_model(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=device,
            name=args.name,
            patience=args.patience
        )
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
