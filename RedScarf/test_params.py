#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数调整测试工具
用于快速测试不同的参数组合，找到最适合的配置
"""

import cv2
import numpy as np
from pathlib import Path
from detection_service import RedScarfDetectionService
from detector.utils import is_wearing_redscarf, calculate_iou
from ultralytics import YOLO
import sys


class ParameterTester:
    """参数测试工具"""
    
    def __init__(self):
        """初始化检测器"""
        print("🔧 初始化检测器...")
        self.detector = RedScarfDetectionService()
    
    def test_image_with_params(self, image_path: str, iou_threshold: float, 
                               vertical_ratio: float, verbose: bool = True):
        """
        用指定参数检测图片
        
        Args:
            image_path: 图片路径
            iou_threshold: IoU阈值
            vertical_ratio: 垂直位置比例
            verbose: 是否打印详细信息
        
        Returns:
            (wearing_count, not_wearing_count, person_count): 佩戴/未佩戴/总人数
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测红领巾
        redscarf_results = self.detector.redscarf_model(image_rgb, verbose=False)
        redscarf_boxes = []
        
        for result in redscarf_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= 0.55:
                    redscarf_boxes.append(box.xyxy[0].tolist())
        
        # 检测人体
        person_results = self.detector.person_model(image_rgb, verbose=False)
        
        person_count = 0
        wearing_count = 0
        not_wearing_count = 0
        
        for result in person_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == 0 and conf >= 0.5:
                    person_count += 1
                    xyxy = box.xyxy[0].tolist()
                    
                    # 使用指定参数判断
                    is_wearing, _ = is_wearing_redscarf(
                        np.array(xyxy), redscarf_boxes,
                        iou_threshold=iou_threshold,
                        vertical_ratio_threshold=vertical_ratio
                    )
                    
                    if is_wearing:
                        wearing_count += 1
                    else:
                        not_wearing_count += 1
        
        if verbose:
            print(f"   👥 人体: {person_count} | "
                  f"✅ 已佩戴: {wearing_count} | "
                  f"❌ 未佩戴: {not_wearing_count}")
        
        return wearing_count, not_wearing_count, person_count
    
    def compare_parameters(self, image_path: str):
        """
        对比不同参数组合的效果
        
        Args:
            image_path: 图片路径
        """
        print("\n" + "="*80)
        print("📊 参数对比测试")
        print("="*80)
        print(f"图片: {image_path}\n")
        
        # 定义参数组合
        params = [
            ("严格 (高准确)", 0.2, 0.4),
            ("较严格", 0.15, 0.45),
            ("平衡 (推荐)", 0.1, 0.55),
            ("较宽松", 0.08, 0.6),
            ("宽松 (高漏检)", 0.05, 0.7),
        ]
        
        print(f"{'配置名称':<15} {'IoU':<7} {'垂直比':<7} {'人体':<5} {'已佩':<5} {'未佩':<5} {'正确率':<8}")
        print("-"*80)
        
        for name, iou_thresh, vert_ratio in params:
            result = self.test_image_with_params(
                image_path, iou_thresh, vert_ratio, verbose=False
            )
            
            if result:
                wearing, not_wearing, total = result
                if total > 0:
                    print(f"{name:<15} {iou_thresh:<7.2f} {vert_ratio:<7.2f} "
                          f"{total:<5} {wearing:<5} {not_wearing:<5}", end="")
                    
                    # 这里需要用户告诉我们正确答案
                    print()
    
    def interactive_test(self, image_path: str):
        """
        交互式测试 - 用户设定目标，系统找最优参数
        
        Args:
            image_path: 图片路径
        """
        print("\n" + "="*80)
        print("🎯 交互式参数调整")
        print("="*80)
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            return
        
        print(f"图片: {image_path}")
        print("\n请告诉我这张图片中有多少人已佩戴红领巾:")
        
        try:
            expected_wearing = int(input("已佩戴红领巾的人数: "))
            expected_total = int(input("总人数: "))
        except ValueError:
            print("❌ 输入无效")
            return
        
        expected_not_wearing = expected_total - expected_wearing
        
        print(f"\n目标: 检测出 {expected_wearing}/{expected_total} 人已佩戴")
        print("\n开始搜索最优参数...\n")
        
        best_params = None
        best_error = float('inf')
        
        # 网格搜索
        iou_values = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
        vert_values = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        
        results = []
        
        for iou in iou_values:
            for vert in vert_values:
                result = self.test_image_with_params(
                    image_path, iou, vert, verbose=False
                )
                
                if result:
                    wearing, not_wearing, total = result
                    
                    # 计算误差
                    wearing_error = abs(wearing - expected_wearing)
                    not_wearing_error = abs(not_wearing - expected_not_wearing)
                    total_error = wearing_error + not_wearing_error
                    
                    results.append({
                        'iou': iou,
                        'vert': vert,
                        'wearing': wearing,
                        'error': total_error
                    })
                    
                    if total_error < best_error:
                        best_error = total_error
                        best_params = (iou, vert, wearing, not_wearing)
        
        # 显示最优结果
        if best_params:
            iou, vert, wearing, not_wearing = best_params
            print("\n" + "="*80)
            print("✅ 最优参数找到！")
            print("="*80)
            print(f"IoU阈值: {iou}")
            print(f"垂直比例: {vert}")
            print(f"检测结果: {wearing} 人已佩戴, {not_wearing} 人未佩戴")
            print(f"误差: {best_error}")
            print(f"\n建议在 config.py 中设置:")
            print(f"  REDSCARF_IOU_THRESHOLD = {iou}")
            print(f"  REDSCARF_VERTICAL_RATIO = {vert}")
        
        # 显示 top-5 结果
        results.sort(key=lambda x: x['error'])
        print(f"\n📈 Top-5 最优参数组合:")
        print("-"*80)
        print(f"{'排名':<5} {'IoU':<7} {'垂直比':<7} {'检测':<5} {'误差':<7}")
        print("-"*80)
        
        for i, res in enumerate(results[:5], 1):
            print(f"{i:<5} {res['iou']:<7.2f} {res['vert']:<7.2f} "
                  f"{res['wearing']:<5} {res['error']:<7}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  1. 对比参数效果:")
        print("     python test_params.py compare image.jpg")
        print("  2. 交互式调参:")
        print("     python test_params.py interactive image.jpg")
        return
    
    mode = sys.argv[1]
    
    if len(sys.argv) < 3:
        print("❌ 请指定图片路径")
        return
    
    image_path = sys.argv[2]
    
    tester = ParameterTester()
    
    if mode == "compare":
        tester.compare_parameters(image_path)
    elif mode == "interactive":
        tester.interactive_test(image_path)
    else:
        print(f"❌ 未知模式: {mode}")


if __name__ == "__main__":
    main()
