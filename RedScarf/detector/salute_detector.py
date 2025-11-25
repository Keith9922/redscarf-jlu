"""
敬礼检测模块 - 基于关键点和手掌检测的敬礼姿态识别
用于判断小学生敬礼动作是否标准
"""
import numpy as np
from typing import Dict, Tuple, Optional
import math
import cv2


class SaluteDetector:
    """敬礼检测器 - 基于关键点算法"""
    
    def __init__(self, 
                 angle_threshold: Tuple[float, float] = (60, 120),
                 hand_head_distance_ratio: float = 0.3,
                 strict_mode: bool = False,
                 hand_openness_threshold: float = 0.35):
        """
        初始化敬礼检测器
        
        Args:
            angle_threshold: 手肘角度范围 (最小角度, 最大角度)，单位：度
            hand_head_distance_ratio: 手部到头部的距离比例阈值
            strict_mode: 是否使用严格模式
            hand_openness_threshold: 手部开放度阈值（<0.35表示五指并拢）
        """
        self.angle_threshold = angle_threshold
        self.hand_head_distance_ratio = hand_head_distance_ratio
        self.strict_mode = strict_mode
        self.hand_openness_threshold = hand_openness_threshold
    
    def detect_salute(self, keypoints: np.ndarray, image: Optional[np.ndarray] = None) -> Dict:
        """
        检测敬礼姿态
        
        Args:
            keypoints: 关键点数组 [17, 3] (x, y, confidence)
            image: 输入图像（可选，用于手掌检测），BGR格式
        
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
                    'posture': str,             # 整体姿态
                    'hand_openness': float,     # 手部开放度
                    'palm_quality': str         # 手掌质量评价
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
            side='left',
            image=image
        )
        
        # 检测右手敬礼
        right_salute = self._check_single_hand_salute(
            wrist=right_wrist,
            elbow=right_elbow,
            shoulder=right_shoulder,
            head_center=nose,
            ear=right_ear,
            side='right',
            image=image
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
                                   ear: np.ndarray, side: str, 
                                   image: Optional[np.ndarray] = None) -> Dict:
        """
        检查单手敬礼姿态
        
        Args:
            wrist: 手腕关键点
            elbow: 手肘关键点
            shoulder: 肩膀关键点
            head_center: 头部中心点（鼻子）
            ear: 耳朵关键点
            side: 检测侧 ('left' 或 'right')
            image: 输入图像（可选）
        
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
        
        # 新增：手掌检测（可选，需要图像）
        hand_openness = None
        palm_quality_score = 0
        if image is not None:
            hand_roi, roi_offset = self._extract_hand_roi(
                image, wrist, elbow, shoulder
            )
            if hand_roi is not None and hand_roi.size > 0:
                openness_result = self._check_hand_openness(hand_roi)
                if openness_result is not None:
                    hand_openness = openness_result['openness']
                    is_closed = openness_result['is_closed']
                    
                    # 五指并拢评分 (20分)
                    if is_closed:
                        palm_quality_score = 20
                        score += 20
                    else:
                        # 根据开放度线性减分
                        penalty = openness_result['openness'] * 20
                        palm_quality_score = max(0, 20 - penalty)
                        score += palm_quality_score
        
        # 更新结果
        result['score'] = min(100, score)
        result['side'] = side if score >= 60 else 'none'
        result['details']['hand_openness'] = hand_openness if hand_openness is not None else 0.0
        
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
        
        # 手掌质量评价
        if hand_openness is not None:
            if hand_openness < self.hand_openness_threshold:
                result['details']['palm_quality'] = '优秀 - 五指并拢 ✅'
            elif hand_openness < self.hand_openness_threshold + 0.15:
                result['details']['palm_quality'] = '良好 - 手指基本并拢'
            else:
                result['details']['palm_quality'] = '一般 - 手指张开过大'
        
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
    
    def _extract_hand_roi(self, image: np.ndarray, wrist: np.ndarray, 
                         elbow: np.ndarray, shoulder: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        从图像中提取手部区域ROI
        
        Args:
            image: 输入图像
            wrist: 手腕关键点
            elbow: 手肘关键点
            shoulder: 肩膀关键点
        
        Returns:
            (hand_roi, roi_offset): 手部区域和偏移量
        """
        # 计算手臂长度（肘到手腕的距离）
        arm_length = np.linalg.norm(elbow[:2] - wrist[:2])
        
        if arm_length < 10:  # 手臂长度太小，可能是坏数据
            return None, (0, 0)
        
        # 手部ROI应该是手臂长度的0.4-0.6倍
        roi_size = arm_length * 0.5
        
        # 提取ROI区域（正方形）
        x1 = max(0, int(wrist[0] - roi_size))
        y1 = max(0, int(wrist[1] - roi_size))
        x2 = min(image.shape[1], int(wrist[0] + roi_size))
        y2 = min(image.shape[0], int(wrist[1] + roi_size))
        
        if x1 >= x2 or y1 >= y2:
            return None, (0, 0)
        
        hand_roi = image[y1:y2, x1:x2]
        return hand_roi, (x1, y1)
    
    def _check_hand_openness(self, hand_roi: np.ndarray, 
                            skin_threshold: float = 0.15) -> Optional[Dict]:
        """
        计算手部开放度（Five-Finger Openness）
        
        原理：
        - 使用肤色识别提取手部轮廓
        - 计算凸包面积与实际手部面积比
        - 开放度低 = 五指并拢 ✓
        
        Args:
            hand_roi: 手部ROI区域
            skin_threshold: 肤色阈值
        
        Returns:
            检测结果字典或None
        """
        try:
            if hand_roi.size == 0:
                return None
            
            # 肤色范围定义 (HSV颜色空间)
            # 扩展范围以覆盖不同肤色
            lower_skin1 = np.array([0, 10, 60])
            upper_skin1 = np.array([20, 255, 255])
            lower_skin2 = np.array([160, 10, 60])
            upper_skin2 = np.array([180, 255, 255])
            
            hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
            
            # 创建肤色掩码
            mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(mask1, mask2)
            
            # 形态学操作，去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # 获取手部轮廓
            contours, _ = cv2.findContours(
                skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return None
            
            # 最大轮廓应该是手部
            hand_contour = max(contours, key=cv2.contourArea)
            hand_area = cv2.contourArea(hand_contour)
            
            if hand_area < 50:  # 轮廓太小
                return None
            
            # 计算凸包
            hull = cv2.convexHull(hand_contour)
            hull_area = cv2.contourArea(hull)
            
            # 开放度 = (凸包面积 - 手部面积) / 凸包面积
            # 值越小表示手指越并拢
            if hull_area == 0:
                openness = 0.0
            else:
                openness = (hull_area - hand_area) / hull_area
            
            return {
                'openness': openness,
                'is_closed': openness < self.hand_openness_threshold,
                'hand_contour': hand_contour,
                'hull': hull,
                'hand_area': hand_area,
                'hull_area': hull_area
            }
        
        except Exception as e:
            print(f"[WARNING] 手部检测失败: {e}")
            return None
    
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
        
        # 添加手掌质量反馈
        if details.get('hand_openness', 0) > 0:
            feedback += f"\n- 手指状态: {details.get('palm_quality', '未检测')}\n"
            feedback += f"  (开放度: {details['hand_openness']:.2f})"
        
        feedback += f"\n- 整体评价: {details['posture']}"
        
        return feedback
