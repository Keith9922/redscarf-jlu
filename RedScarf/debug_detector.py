#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检测调试脚本 - 用于诊断检测问题
帮你了解为什么某些佩戴了红领巾的人没有被检测出来
"""

import cv2
import numpy as np
from pathlib import Path
from detector.utils import is_wearing_redscarf, calculate_iou
from detection_service import RedScarfDetectionService
from config import (
    REDSCARF_IOU_THRESHOLD, REDSCARF_VERTICAL_RATIO,
    COLOR_WEARING_REDSCARF, COLOR_NOT_WEARING, COLOR_REDSCARF_BOX
)


def debug_detection(image_path: str):
    """
    调试图片检测，输出详细信息
    
    Args:
        image_path: 图片路径
    """
    print("=" * 80)
    print("🔍 红领巾检测调试工具")
    print("=" * 80)
    print()
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    print(f"📷 图片文件: {image_path}")
    print(f"   分辨率: {image.shape[1]}x{image.shape[0]}")
    print()
    
    # 初始化检测器
    try:
        detector = RedScarfDetectionService()
    except Exception as e:
        print(f"❌ 初始化检测器失败: {e}")
        return
    
    # 检测人体和红领巾
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("🔎 开始检测...")
    print()
    
    # 检测红领巾
    print("1️⃣  红领巾检测结果:")
    print("-" * 80)
    
    redscarf_results = detector.redscarf_model(image_rgb, verbose=False)
    redscarf_boxes = []
    
    redscarf_count = 0
    for result in redscarf_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= 0.3:  # 降低阈值以显示所有可能的红领巾
                redscarf_count += 1
                xyxy = box.xyxy[0].tolist()
                redscarf_boxes.append(xyxy)
                
                x1, y1, x2, y2 = xyxy
                center_y = (y1 + y2) / 2
                
                print(f"   红领巾 #{redscarf_count}:")
                print(f"     坐标: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
                print(f"     中心Y: {center_y:.1f}")
                print(f"     置信度: {conf:.2%}")
                print(f"     是否超过阈值(0.55): {'✅' if conf >= 0.55 else '❌'}")
                print()
    
    if redscarf_count == 0:
        print("   ❌ 没有检测到红领巾")
        print()
    
    # 检测人体
    print("2️⃣  人体检测结果:")
    print("-" * 80)
    
    person_results = detector.person_model(image_rgb, verbose=False)
    person_count = 0
    
    for result in person_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if cls == 0 and conf >= 0.3:  # 降低阈值
                person_count += 1
                xyxy = box.xyxy[0].tolist()
                
                x1, y1, x2, y2 = xyxy
                height = y2 - y1
                width = x2 - x1
                
                print(f"   人体 #{person_count}:")
                print(f"     坐标: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
                print(f"     大小: {width:.1f}x{height:.1f}")
                print(f"     置信度: {conf:.2%}")
                print(f"     是否超过阈值(0.5): {'✅' if conf >= 0.5 else '❌'}")
                print()
                
                # 检查与红领巾的匹配
                print(f"     🔗 与红领巾的匹配分析:")
                print(f"        -" * 40)
                
                if not redscarf_boxes:
                    print(f"        没有有效的红领巾可以匹配")
                else:
                    for i, redscarf_box in enumerate(redscarf_boxes):
                        rx1, ry1, rx2, ry2 = redscarf_box
                        
                        # 计算关键指标
                        iou = calculate_iou(np.array(xyxy), np.array(redscarf_box))
                        
                        redscarf_center_y = (ry1 + ry2) / 2
                        redscarf_center_x = (rx1 + rx2) / 2
                        
                        # 检查位置关系
                        valid_y_min = y1 - height * 0.2
                        valid_y_max = y1 + height * REDSCARF_VERTICAL_RATIO
                        valid_x_min = x1 - width * 0.3
                        valid_x_max = x2 + width * 0.3
                        
                        vertical_in_range = valid_y_min <= redscarf_center_y <= valid_y_max
                        horizontal_in_range = valid_x_min <= redscarf_center_x <= valid_x_max
                        has_horizontal_overlap = not (rx2 < x1 or rx1 > x2)
                        
                        print(f"        📍 红领巾 #{i+1}:")
                        print(f"           IoU: {iou:.4f} (阈值: {REDSCARF_IOU_THRESHOLD})")
                        print(f"           IoU检查: {'✅' if iou > REDSCARF_IOU_THRESHOLD else '❌'}")
                        print()
                        print(f"           Y坐标检查 (范围: {valid_y_min:.1f} ~ {valid_y_max:.1f}):")
                        print(f"           红领巾Y中心: {redscarf_center_y:.1f}")
                        print(f"           Y范围检查: {'✅' if vertical_in_range else '❌'}")
                        print()
                        print(f"           X坐标检查 (范围: {valid_x_min:.1f} ~ {valid_x_max:.1f}):")
                        print(f"           红领巾X中心: {redscarf_center_x:.1f}")
                        print(f"           X范围检查: {'✅' if horizontal_in_range else '❌'}")
                        print(f"           水平重叠检查: {'✅' if has_horizontal_overlap else '❌'}")
                        print()
                
                # 最终判断
                is_wearing, _ = is_wearing_redscarf(
                    np.array(xyxy), redscarf_boxes,
                    iou_threshold=REDSCARF_IOU_THRESHOLD,
                    vertical_ratio_threshold=REDSCARF_VERTICAL_RATIO
                )
                
                print(f"     ✅ 最终判断: {'已佩戴红领巾' if is_wearing else '未佩戴红领巾'}")
                print()
                print()
    
    if person_count == 0:
        print("   ❌ 没有检测到人体")
        print()
    
    # 总结
    print("=" * 80)
    print("📊 检测总结:")
    print(f"   - 检测到的人体: {person_count} 个")
    print(f"   - 检测到的红领巾: {redscarf_count} 个")
    print()
    
    # 建议
    print("💡 调试建议:")
    print(f"   - 当前IoU阈值: {REDSCARF_IOU_THRESHOLD} (范围: 0.0-1.0)")
    print(f"   - 当前垂直位置比例: {REDSCARF_VERTICAL_RATIO}")
    print()
    print("   如果红领巾检测不到:")
    print("   1. 检查红领巾的置信度是否低于0.55")
    print("   2. 尝试降低config.py中的REDSCARF_CONF_THRESHOLD")
    print()
    print("   如果明明有红领巾但判断为未佩戴:")
    print("   1. 尝试降低config.py中的REDSCARF_IOU_THRESHOLD (如0.05)")
    print("   2. 尝试增大config.py中的REDSCARF_VERTICAL_RATIO (如0.6)")
    print("=" * 80)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python debug_detector.py <image_path>")
        print("示例: python debug_detector.py test.jpg")
        return
    
    image_path = sys.argv[1]
    debug_detection(image_path)


if __name__ == "__main__":
    main()
