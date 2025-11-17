"""
敬礼检测模块 - 基于关键点的敬礼姿态识别
用于判断小学生敬礼动作是否标准
"""
import numpy as np
from typing import Dict, Tuple, Optional
import math


class SaluteDetector:
    """敬礼检测器 - 基于关键点算法"""
    
    def __init__(self, 
                 angle_threshold: Tuple[float, float] = (60, 120),
                 hand_head_distance_ratio: float = 0.3,
                 strict_mode: bool = False):
        """
        初始化敬礼检测器
        
        Args:
            angle_threshold: 手肘角度范围 (最小角度, 最大角度)，单位：度
            hand_head_distance_ratio: 手部到头部的距离比例阈值
            strict_mode: 是否使用严格模式
        """
        self.angle_threshold = angle_threshold
        self.hand_head_distance_ratio = hand_head_distance_ratio
        self.strict_mode = strict_mode
    
    def detect_salute(self, keypoints: np.ndarray) -> Dict:
        """
        检测敬礼姿态
        
        Args:
            keypoints: 关键点数组 [17, 3] (x, y, confidence)
        
        Returns:
            检测结果字典:
            {
                'is_saluting': bool,           # 是否在敬礼
                'side': str,                    # 敬礼手侧 ('left', 'right', 'none')
                'score': float,                 # 姿态得分 (0-100)
                'details': {                    # 详细信息
                    'hand_position': str,       # 手部位置
                    'elbow_angle': float,       # 手肘角度
                    'hand_height': str,         # 手部高度
                    'posture': str              # 整体姿态
                }
            }
        """
        # 提取关键点
        nose = keypoints[0]            # 鼻子
        left_eye = keypoints[1]        # 左眼
        right_eye = keypoints[2]       # 右眼
        left_ear = keypoints[3]        # 左耳
        right_ear = keypoints[4]       # 右耳
        left_shoulder = keypoints[5]   # 左肩
        right_shoulder = keypoints[6]  # 右肩
        left_elbow = keypoints[7]      # 左肘
        right_elbow = keypoints[8]     # 右肘
        left_wrist = keypoints[9]      # 左手腕
        right_wrist = keypoints[10]    # 右手腕
        
        # 初始化结果
        result = {
            'is_saluting': False,
            'side': 'none',
            'score': 0.0,
            'details': {
                'hand_position': '未检测到',
                'elbow_angle': 0.0,
                'hand_height': '未知',
                'posture': '未敬礼'
            }
        }
        
        # 检测左手敬礼
        left_salute = self._check_single_hand_salute(
            wrist=left_wrist,
            elbow=left_elbow,
            shoulder=left_shoulder,
            head_center=nose,
            ear=left_ear,
            side='left'
        )
        
        # 检测右手敬礼
        right_salute = self._check_single_hand_salute(
            wrist=right_wrist,
            elbow=right_elbow,
            shoulder=right_shoulder,
            head_center=nose,
            ear=right_ear,
            side='right'
        )
        
        # 选择得分更高的一侧
        if left_salute['score'] > right_salute['score']:
            result = left_salute
        else:
            result = right_salute
        
        # 判断是否敬礼
        min_score = 70 if self.strict_mode else 60
        result['is_saluting'] = result['score'] >= min_score
        
        return result
    
    def _check_single_hand_salute(self, wrist: np.ndarray, elbow: np.ndarray,
                                   shoulder: np.ndarray, head_center: np.ndarray,
                                   ear: np.ndarray, side: str) -> Dict:
        """
        检查单手敬礼姿态
        
        Args:
            wrist: 手腕关键点
            elbow: 手肘关键点
            shoulder: 肩膀关键点
            head_center: 头部中心点（鼻子）
            ear: 耳朵关键点
            side: 检测侧 ('left' 或 'right')
        
        Returns:
            检测结果
        """
        result = {
            'is_saluting': False,
            'side': 'none',
            'score': 0.0,
            'details': {
                'hand_position': '未检测到',
                'elbow_angle': 0.0,
                'hand_height': '未知',
                'posture': '未敬礼'
            }
        }
        
        # 检查关键点置信度
        if wrist[2] < 0.5 or elbow[2] < 0.5 or shoulder[2] < 0.5 or head_center[2] < 0.5:
            return result
        
        # 1. 计算手肘角度
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        result['details']['elbow_angle'] = elbow_angle
        
        # 2. 检查手部高度（应该在头部附近）
        hand_y = wrist[1]
        head_y = head_center[1]
        ear_y = ear[1] if ear[2] > 0.3 else head_y
        
        # 计算头部高度范围
        head_height_range = abs(head_y - shoulder[1])
        
        # 手部应该在头部高度的合理范围内
        hand_above_shoulder = hand_y < shoulder[1]
        hand_near_head = abs(hand_y - head_y) < head_height_range * 0.8
        
        # 3. 检查手部横向位置（应该在头部附近）
        hand_x = wrist[0]
        head_x = head_center[0]
        shoulder_x = shoulder[0]
        
        # 计算头部到肩膀的距离
        head_shoulder_dist = abs(head_x - shoulder_x)
        
        # 手部应该在头部横向范围内
        hand_near_head_horizontal = abs(hand_x - head_x) < head_shoulder_dist * 2
        
        # 4. 评分系统
        score = 0.0
        
        # 手肘角度评分 (30分)
        if self.angle_threshold[0] <= elbow_angle <= self.angle_threshold[1]:
            # 最佳角度在90度左右
            angle_score = 30 - abs(elbow_angle - 90) * 0.3
            score += max(0, angle_score)
        
        # 手部高度评分 (35分)
        if hand_above_shoulder:
            if hand_near_head:
                score += 35
            else:
                score += 20
        
        # 手部横向位置评分 (25分)
        if hand_near_head_horizontal:
            score += 25
        
        # 整体姿态评分 (10分)
        # 检查手肘是否抬起
        elbow_raised = elbow[1] < shoulder[1]
        if elbow_raised:
            score += 10
        
        # 更新结果
        result['score'] = min(100, score)
        result['side'] = side if score >= 60 else 'none'
        
        # 详细信息
        if hand_above_shoulder and hand_near_head:
            result['details']['hand_height'] = '正确（头部附近）'
        elif hand_above_shoulder:
            result['details']['hand_height'] = '较高'
        else:
            result['details']['hand_height'] = '偏低'
        
        if hand_near_head_horizontal:
            result['details']['hand_position'] = '正确（头部侧方）'
        else:
            result['details']['hand_position'] = '偏离头部'
        
        # 姿态评价
        if score >= 90:
            result['details']['posture'] = '标准敬礼 ✅'
        elif score >= 75:
            result['details']['posture'] = '敬礼姿态良好'
        elif score >= 60:
            result['details']['posture'] = '敬礼姿态基本正确'
        else:
            result['details']['posture'] = '未敬礼或姿态不标准'
        
        return result
    
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, 
                        point3: np.ndarray) -> float:
        """
        计算三个点形成的角度（point2为顶点）
        
        Args:
            point1: 第一个点
            point2: 顶点
            point3: 第三个点
        
        Returns:
            角度值（度）
        """
        # 计算向量
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # 计算向量长度
        len1 = np.linalg.norm(vector1)
        len2 = np.linalg.norm(vector2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 计算夹角余弦值
        cos_angle = np.dot(vector1, vector2) / (len1 * len2)
        
        # 限制在[-1, 1]范围内，避免数值误差
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 计算角度（弧度转角度）
        angle = math.acos(cos_angle) * 180 / math.pi
        
        return angle
    
    def get_salute_feedback(self, result: Dict) -> str:
        """
        生成敬礼反馈文本
        
        Args:
            result: 检测结果
        
        Returns:
            反馈文本
        """
        if not result['is_saluting']:
            return "未检测到敬礼动作或姿态不标准"
        
        score = result['score']
        side = '左手' if result['side'] == 'left' else '右手'
        details = result['details']
        
        feedback = f"{side}敬礼 (得分: {score:.1f}/100)\n"
        feedback += f"- 手肘角度: {details['elbow_angle']:.1f}°\n"
        feedback += f"- 手部位置: {details['hand_position']}\n"
        feedback += f"- 手部高度: {details['hand_height']}\n"
        feedback += f"- 整体评价: {details['posture']}"
        
        return feedback
