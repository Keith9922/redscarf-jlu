#!/usr/bin/env python3
"""
红领巾检测调试脚本 - 诊断红领巾检测问题
可以加载图片，显示所有检测结果（不考虑置信度阈值），帮助诊断问题
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import argparse

# 导入项目配置
from config import (
    REDSCARF_CONF_THRESHOLD, PERSON_CONF_THRESHOLD,
    COLOR_REDSCARF_BOX, COLOR_WEARING_REDSCARF, COLOR_NOT_WEARING,
    BOX_LINE_THICKNESS, FONT_SCALE
)


def draw_boxes(image, boxes, label_prefix="", color=(0, 255, 0)):
    """
    绘制检测框
    
    Args:
        image: 输入图像
        boxes: 检测框列表，每个框为 [x1, y1, x2, y2, conf]
        label_prefix: 标签前缀
        color: 框颜色
    """
    for box_info in boxes:
        if len(box_info) == 5:
            x1, y1, x2, y2, conf = int(box_info[0]), int(box_info[1]), int(box_info[2]), int(box_info[3]), box_info[4]
        else:
            x1, y1, x2, y2 = int(box_info[0]), int(box_info[1]), int(box_info[2]), int(box_info[3])
            conf = 0
        
        # 绘制框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_LINE_THICKNESS)
        
        # 绘制标签
        label = f'{label_prefix} {conf:.3f}' if conf > 0 else label_prefix
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, FONT_SCALE, 1)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), font, FONT_SCALE, (255, 255, 255), 1)
    
    return image


def debug_redscarf_detection(image_path):
    """
    调试红领巾检测
    """
    print("=" * 70)
    print("🔍 红领巾检测详细诊断")
    print("=" * 70)
    
    # 检查图像文件
    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    print(f"\n📂 加载图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法加载图像")
        return
    
    print(f"  ✓ 图像尺寸: {image.shape[1]}x{image.shape[0]}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 加载红领巾模型
    print(f"\n🤖 加载红领巾检测模型...")
    redscarf_model_path = 'data/models/redscarf.pt'
    try:
        redscarf_model = YOLO(redscarf_model_path)
        print(f"  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return
    
    # 加载人体检测模型
    print(f"\n🤖 加载人体检测模型...")
    try:
        person_model = YOLO('../yolov8n.pt')
        print(f"  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return
    
    # 检测红领巾 - 所有置信度
    print(f"\n🔍 检测红领巾 (所有置信度)...")
    redscarf_results = redscarf_model(image_rgb, verbose=False)
    all_redscarf_boxes = []
    filtered_redscarf_boxes = []
    
    if len(redscarf_results) > 0:
        result = redscarf_results[0]
        print(f"  ✓ 检测到 {len(result.boxes)} 个红领巾候选")
        
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            all_redscarf_boxes.append(xyxy + [conf])
            
            if conf >= REDSCARF_CONF_THRESHOLD:
                filtered_redscarf_boxes.append(xyxy)
                status = "✓ PASS (保留)"
            else:
                status = f"✗ FAIL (置信度{conf:.3f} < {REDSCARF_CONF_THRESHOLD})"
            
            print(f"    [{i+1}] 置信度: {conf:.4f} - {status}")
    else:
        print(f"  ℹ️  未检测到红领巾")
    
    # 检测人体 - 所有置信度
    print(f"\n🔍 检测人体 (所有置信度)...")
    person_results = person_model(image_rgb, verbose=False)
    all_person_boxes = []
    
    if len(person_results) > 0:
        result = person_results[0]
        print(f"  ✓ 检测到 {len(result.boxes)} 个目标")
        
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            
            class_name = "person" if cls == 0 else f"class_{cls}"
            
            if cls == 0:
                all_person_boxes.append(xyxy + [conf])
                if conf >= PERSON_CONF_THRESHOLD:
                    status = "✓ PASS (保留)"
                else:
                    status = f"✗ FAIL (置信度{conf:.3f} < {PERSON_CONF_THRESHOLD})"
                print(f"    [{i+1}] {class_name} - 置信度: {conf:.4f} - {status}")
            else:
                print(f"    [{i+1}] {class_name} - 置信度: {conf:.4f} - ✗ SKIP (非人体类别)")
    else:
        print(f"  ℹ️  未检测到任何目标")
    
    # 绘制所有检测结果
    print(f"\n🎨 绘制检测结果...")
    result_image = image.copy()
    
    # 绘制所有红领巾（浅蓝色）
    for i, box in enumerate(all_redscarf_boxes):
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (200, 100, 0), 1)  # 浅蓝
        cv2.putText(result_image, f'ALL_RS {conf:.3f}', (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)
    
    # 绘制通过阈值的红领巾（青色）
    for box in filtered_redscarf_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), COLOR_REDSCARF_BOX, BOX_LINE_THICKNESS)
        cv2.putText(result_image, 'PASS_RS', (x1, y1-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_REDSCARF_BOX, BOX_LINE_THICKNESS)
    
    # 绘制所有人体（白色虚线）
    for i, box in enumerate(all_person_boxes):
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        # 虚线绘制
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(result_image, f'PERSON {conf:.3f}', (x1, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 添加图例
    cv2.putText(result_image, 'Legend:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(result_image, (10, 40), (150, 60), (200, 100, 0), 1)
    cv2.putText(result_image, 'All Redscarf', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)
    
    cv2.rectangle(result_image, (160, 40), (300, 60), COLOR_REDSCARF_BOX, BOX_LINE_THICKNESS)
    cv2.putText(result_image, 'Passed RS', (170, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_REDSCARF_BOX, 1)
    
    cv2.rectangle(result_image, (310, 40), (450, 60), (255, 255, 255), 1)
    cv2.putText(result_image, 'Person', (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存结果
    output_path = Path(image_path).stem + '_debug.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"  ✓ 结果已保存: {output_path}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("📊 诊断总结")
    print("=" * 70)
    print(f"  红领巾总检测数: {len(all_redscarf_boxes)}")
    print(f"  红领巾通过阈值数: {len(filtered_redscarf_boxes)}")
    print(f"  人体总检测数: {len(all_person_boxes)}")
    print(f"  配置的阈值:")
    print(f"    - REDSCARF_CONF_THRESHOLD: {REDSCARF_CONF_THRESHOLD}")
    print(f"    - PERSON_CONF_THRESHOLD: {PERSON_CONF_THRESHOLD}")
    
    if len(all_redscarf_boxes) == 0:
        print("\n⚠️  问题: 模型完全未检测到红领巾")
        print("   可能原因:")
        print("     1. 图像中没有红领巾")
        print("     2. 红领巾太小或质量太差")
        print("     3. 模型训练不足")
    elif len(filtered_redscarf_boxes) == 0:
        print("\n⚠️  问题: 检测到红领巾但置信度都太低")
        print(f"   建议: 降低REDSCARF_CONF_THRESHOLD (当前: {REDSCARF_CONF_THRESHOLD})")
    else:
        print("\n✅ 检测正常！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='红领巾检测调试工具')
    parser.add_argument('image', nargs='?', help='待检测的图像路径')
    args = parser.parse_args()
    
    if args.image:
        debug_redscarf_detection(args.image)
    else:
        print("用法: python debug_redscarf.py <image_path>")
        print("\n示例:")
        print("  python debug_redscarf.py test.jpg")
