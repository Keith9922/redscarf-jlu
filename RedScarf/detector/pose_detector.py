"""
人体姿态检测模块 - 基于YOLOv8-Pose
用于检测人体关键点和姿态识别
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch


class PoseDetector:
    """YOLOv8-Pose 姿态检测器"""
    
    # YOLOv8-Pose 17个关键点定义
    KEYPOINT_NAMES = [
        'nose',           # 0: 鼻子
        'left_eye',       # 1: 左眼
        'right_eye',      # 2: 右眼
        'left_ear',       # 3: 左耳
        'right_ear',      # 4: 右耳
        'left_shoulder',  # 5: 左肩
        'right_shoulder', # 6: 右肩
        'left_elbow',     # 7: 左肘
        'right_elbow',    # 8: 右肘
        'left_wrist',     # 9: 左手腕
        'right_wrist',    # 10: 右手腕
        'left_hip',       # 11: 左髋
        'right_hip',      # 12: 右髋
        'left_knee',      # 13: 左膝
        'right_knee',     # 14: 右膝
        'left_ankle',     # 15: 左踝
        'right_ankle'     # 16: 右踝
    ]
    
    def __init__(self, model_path: str, device: str = 'cpu', conf_threshold: float = 0.5):
        """
        初始化姿态检测器
        
        Args:
            model_path: YOLOv8-Pose模型路径
            device: 运行设备 ('cpu' 或 'cuda')
            conf_threshold: 置信度阈值
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_gpu = device.lower() == 'gpu' and torch.cuda.is_available()
        
        # 加载YOLOv8-Pose模型
        print(f"[INFO] 正在加载姿态检测模型: {model_path}")
        self.model = YOLO(model_path)
        
        if self.use_gpu:
            self.model.to('cuda')
        
        print(f"[INFO] 姿态检测模型加载完成 (Device: {device}, GPU: {self.use_gpu})")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
        
        Returns:
            检测结果列表，每个结果包含:
            - bbox: 人体边界框 [x1, y1, x2, y2]
            - keypoints: 关键点坐标 [[x, y, conf], ...]
            - conf: 检测置信度
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLOv8-Pose推理
        results = self.model(image_rgb, verbose=False)
        
        detections = []
        
        for result in results:
            # 检查是否有检测结果
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            
            # 遍历每个检测到的人体
            for i in range(len(result.boxes)):
                box = result.boxes[i]
                conf = float(box.conf[0])
                
                # 置信度过滤
                if conf < self.conf_threshold:
                    continue
                
                # 获取边界框
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # 获取关键点 (shape: [17, 3] - x, y, confidence)
                keypoints = result.keypoints[i].data[0].cpu().numpy()
                
                detections.append({
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'conf': conf
                })
        
        return detections
    
    def draw_pose(self, image: np.ndarray, detections: List[Dict], 
                  draw_bbox: bool = True, draw_skeleton: bool = True) -> np.ndarray:
        """
        在图像上绘制姿态检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果
            draw_bbox: 是否绘制边界框
            draw_skeleton: 是否绘制骨架连线
        
        Returns:
            绘制后的图像
        """
        result_image = image.copy()
        
        # 定义骨架连线 (COCO格式)
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12],  # 腿部
            [11, 12],                                 # 髋部
            [5, 11], [6, 12],                        # 躯干
            [5, 6],                                   # 肩部
            [5, 7], [7, 9],                          # 左臂
            [6, 8], [8, 10],                         # 右臂
            [1, 3], [2, 4],                          # 耳朵
            [0, 1], [0, 2]                           # 鼻子和眼睛
        ]
        
        for det in detections:
            bbox = det['bbox']
            keypoints = det['keypoints']
            conf = det['conf']
            
            # 绘制边界框
            if draw_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                label = f'Person {conf:.2f}'
                cv2.putText(result_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # 绘制关键点
            for kp in keypoints:
                x, y, kp_conf = kp
                if kp_conf > 0.5:  # 只绘制置信度高的关键点
                    cv2.circle(result_image, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # 绘制骨架连线
            if draw_skeleton:
                for connection in skeleton:
                    pt1_idx, pt2_idx = connection
                    pt1 = keypoints[pt1_idx]
                    pt2 = keypoints[pt2_idx]
                    
                    # 只有两个关键点置信度都足够高时才绘制连线
                    if pt1[2] > 0.5 and pt2[2] > 0.5:
                        pt1_coords = (int(pt1[0]), int(pt1[1]))
                        pt2_coords = (int(pt2[0]), int(pt2[1]))
                        cv2.line(result_image, pt1_coords, pt2_coords, (0, 255, 255), 2)
        
        return result_image
    
    def get_keypoint(self, keypoints: np.ndarray, name: str) -> Optional[np.ndarray]:
        """
        获取指定名称的关键点
        
        Args:
            keypoints: 关键点数组 [17, 3]
            name: 关键点名称
        
        Returns:
            关键点坐标 [x, y, conf] 或 None
        """
        if name not in self.KEYPOINT_NAMES:
            return None
        
        idx = self.KEYPOINT_NAMES.index(name)
        kp = keypoints[idx]
        
        # 如果置信度太低，返回None
        if kp[2] < 0.3:
            return None
        
        return kp
