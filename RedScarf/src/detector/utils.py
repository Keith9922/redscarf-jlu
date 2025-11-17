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
                        vertical_ratio_threshold: float = 0.6) -> Tuple[bool, Optional[np.ndarray]]:
    """
    判断一个人是否佩戴了红领巾
    
    改进的判断逻辑:
    1. 检查是否存在与人体有重叠的红领巾
    2. 红领巾位置应该在人体的合理区域内(脖子到上半身)
    3. 使用更灵活的匹配策略：
       - 优先匹配IoU最高的红领巾
       - 或者匹配位置最合理的红领巾
    
    Args:
        person_box: 人体边界框 [x1, y1, x2, y2]
        redscarf_boxes: 红领巾边界框列表
        iou_threshold: 最小IoU阈值 (非必须条件)
        vertical_ratio_threshold: 垂直位置比例阈值(0-1，表示有效区域占人体高度的比例)
    
    Returns:
        (is_wearing, matched_redscarf): 是否佩戴红领巾和匹配的红领巾框
    """
    if not redscarf_boxes:
        return False, None
    
    px1, py1, px2, py2 = person_box
    person_height = py2 - py1
    person_width = px2 - px1
    
    # 定义红领巾应该出现的有效区域
    # 从人体上方到人体高度的60%处（包括脖子、胸部区域）
    valid_y_min = py1 - person_height * 0.2  # 允许红领巾在人体上方
    valid_y_max = py1 + person_height * vertical_ratio_threshold
    
    # 定义水平有效区域（人体宽度范围）
    valid_x_min = px1 - person_width * 0.2
    valid_x_max = px2 + person_width * 0.2
    
    best_match = None
    best_score = 0
    
    for redscarf_box in redscarf_boxes:
        rx1, ry1, rx2, ry2 = redscarf_box
        
        # 计算红领巾中心点
        redscarf_center_x = (rx1 + rx2) / 2
        redscarf_center_y = (ry1 + ry2) / 2
        
        # 计算IoU
        iou = calculate_iou(person_box, redscarf_box)
        
        # 检查位置合理性
        # 条件1: 垂直方向在有效范围内
        vertical_in_range = valid_y_min <= redscarf_center_y <= valid_y_max
        # 条件2: 水平方向在有效范围内
        horizontal_in_range = valid_x_min <= redscarf_center_x <= valid_x_max
        
        # 条件3: 红领巾与人体有水平方向的重叠（宽松条件）
        has_horizontal_overlap = not (rx2 < px1 or rx1 > px2)
        
        # 严格的匹配策略
        # 核心原则：红领巾必须位于人体上半身，且位置合理
        # 只有同时满足以下两个条件才被认为是有效的佩戴：
        # 1. 垂直位置在合理范围（脖子到上半身）
        # 2. 要么IoU足够大，要么水平位置也在范围内
        
        # 计算得分
        current_score = 0
        
        # 只有垂直位置在合理范围内才考虑
        if vertical_in_range:
            # 情况1：IoU足够大 -> 最优
            if iou > iou_threshold:
                position_score = 1.0 - (redscarf_center_y - py1) / person_height
                current_score = iou * 0.7 + position_score * 0.3
            # 情况2：位置合理且有水平重叠 -> 次优（但需要最小IoU）
            elif horizontal_in_range and iou > 0.01:  # 需要至少有微小的重叠
                position_score = 1.0 - (redscarf_center_y - py1) / person_height
                current_score = iou * 0.5 + position_score * 0.3 + 0.2  # 位置合理的基础分
            # 情况3：只有水平重叠，IoU很小 -> 不认为是有效的佩戴
            # 这避免了背景中的红色物体被误判
        
        # 选择最佳匹配
        if current_score > best_score:
            best_score = current_score
            best_match = redscarf_box
    
    # 返回结果：只有当最佳匹配得分足够高时才认为佩戴了红领巾
    # 设定一个得分阈值，防止弱匹配被当作有效佩戴
    is_wearing = best_match is not None and best_score > 0.15
    
    return is_wearing, best_match


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
