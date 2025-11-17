"""
检测工具函数模块
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个边界框的IoU (Intersection over Union)
    
    Args:
        box1: [x1, y1, x2, y2] 格式的边界框
        box2: [x1, y1, x2, y2] 格式的边界框
    
    Returns:
        float: IoU值，范围[0, 1]
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    if union == 0:
        return 0.0
    return intersection / union


def is_wearing_redscarf(person_box: np.ndarray, redscarf_boxes: list, 
                        iou_threshold: float = 0.05,
                        vertical_ratio_threshold: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
    """
    判断一个人是否佩戴了红领巾
    
    简化的判断逻辑（参考RedScarf_back）：
    检查任何红领巾框的关键点是否位于人体框内部
    - 判断红领巾四个角落是否有点在人体框内
    - 检查红领巾是否在人体的合理位置（脖子到上半身）
    
    Args:
        person_box: 人体边界框 [x1, y1, x2, y2]
        redscarf_boxes: 红领巾边界框列表
        iou_threshold: 最小IoU阈值 (未使用，保留兼容性)
        vertical_ratio_threshold: 垂直位置比例阈值(0-1，表示有效区域占人体高度的比例)
    
    Returns:
        (is_wearing, matched_redscarf): 是否佩戴红领巾和匹配的红领巾框
    """
    if not redscarf_boxes:
        return False, None
    
    px1, py1, px2, py2 = person_box
    person_height = py2 - py1
    person_width = px2 - px1
    
    # 定义红领巾有效区域（脖子到胸部）
    valid_y_min = py1
    valid_y_max = py1 + person_height * vertical_ratio_threshold  # 占人体高度的50%
    
    # 定义水平有效区域
    valid_x_min = px1
    valid_x_max = px2
    
    # 遍历所有红领巾，使用RedScarf_back的方式：检查是否有任意点在人体框内
    for redscarf_box in redscarf_boxes:
        rx1, ry1, rx2, ry2 = redscarf_box
        
        # 获取红领巾的四个角点
        redscarf_points = [
            [int(rx1), int(ry1)],  # 左上
            [int(rx1), int(ry2)],  # 左下
            [int(rx2), int(ry1)],  # 右上
            [int(rx2), int(ry2)],  # 右下
        ]
        
        # 检查是否有任何红领巾点在人体框内（RedScarf_back的isUnion逻辑）
        for point in redscarf_points:
            px, py = point
            # 检查点是否在人体框内
            if px1 < px < px2 and py1 < py < py2:
                # 找到了在人体框内的点，再检查垂直位置是否合理
                redscarf_center_y = (ry1 + ry2) / 2
                if valid_y_min <= redscarf_center_y <= valid_y_max:
                    return True, redscarf_box
        
        # 备选方案：如果没有完全在框内的点，检查红领巾中心是否在有效范围内
        # 这处理了红领巾被人体框边缘切割的情况
        redscarf_center_x = (rx1 + rx2) / 2
        redscarf_center_y = (ry1 + ry2) / 2
        
        if (valid_x_min <= redscarf_center_x <= valid_x_max and 
            valid_y_min <= redscarf_center_y <= valid_y_max):
            # 检查是否有水平方向的重叠
            has_horizontal_overlap = not (rx2 < px1 or rx1 > px2)
            if has_horizontal_overlap:
                return True, redscarf_box
    
    return False, None


def draw_detection_box(image: np.ndarray, box: np.ndarray, 
                       label: str, color: Tuple[int, int, int],
                       line_thickness: int = 2,
                       font_scale: float = 0.6) -> np.ndarray:
    """
    在图像上绘制检测框和标签
    
    Args:
        image: 输入图像
        box: 边界框 [x1, y1, x2, y2]
        label: 标签文本
        color: 颜色 (B, G, R)
        line_thickness: 线条粗细
        font_scale: 字体大小
    
    Returns:
        绘制后的图像
    """
    x1, y1, x2, y2 = map(int, box)
    
    # 绘制半透明填充矩形
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
    
    # 绘制边框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
    
    # 绘制标签背景
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = max(line_thickness - 1, 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 标签背景矩形
        label_y1 = max(y1 - text_height - baseline - 5, 0)
        label_y2 = y1
        cv2.rectangle(image, (x1, label_y1), (x1 + text_width + 5, label_y2), 
                     color, -1, cv2.LINE_AA)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1 + 2, y1 - baseline - 2), 
                   font, font_scale, (255, 255, 255), 
                   font_thickness, cv2.LINE_AA)
    
    return image


def draw_fps(image: np.ndarray, fps: float, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    在图像上绘制FPS信息
    
    Args:
        image: 输入图像
        fps: FPS值
        color: 文字颜色
    
    Returns:
        绘制后的图像
    """
    fps_text = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 计算文本位置(图像中心上方)
    (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
    x = (image.shape[1] - text_width) // 2
    y = 30
    
    # 绘制文字背景
    cv2.rectangle(image, (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + 5), (255, 255, 255), -1)
    
    # 绘制文字
    cv2.putText(image, fps_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image
