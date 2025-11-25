"""
红领巾检测服务类 - 封装所有检测逻辑
"""
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from ultralytics import YOLO

from detector.utils import is_wearing_redscarf, draw_detection_box, draw_fps
from detector.pose_detector import PoseDetector
from detector.salute_detector import SaluteDetector
from config import (
    PERSON_MODEL_PATH, REDSCARF_MODEL_PATH, POSE_MODEL_PATH, DEVICE,
    PERSON_CONF_THRESHOLD, REDSCARF_CONF_THRESHOLD, POSE_CONF_THRESHOLD,
    COLOR_WEARING_REDSCARF, COLOR_NOT_WEARING, COLOR_REDSCARF_BOX,
    COLOR_SALUTING, COLOR_NOT_SALUTING,
    BOX_LINE_THICKNESS, FONT_SCALE, DISPLAY_FPS,
    REDSCARF_IOU_THRESHOLD, REDSCARF_VERTICAL_RATIO,
    SALUTE_ANGLE_MIN, SALUTE_ANGLE_MAX, SALUTE_HAND_HEAD_RATIO, SALUTE_STRICT_MODE
)


class RedScarfDetectionService:
    """红领巾检测服务（含敬礼姿态识别）"""
    
    def __init__(self, device: str = DEVICE, enable_pose: bool = True):
        """
        初始化检测服务
        
        Args:
            device: 运行设备 ("CPU" 或 "GPU")
            enable_pose: 是否启用姿态检测功能
        """
        self.device = device
        self.use_gpu = device.upper() == 'GPU' and torch.cuda.is_available()
        self.enable_pose = enable_pose
        
        # 加载模型
        self._load_models()
        
        print(f"[INFO] 红领巾检测服务初始化完成 (Device: {device}, GPU: {self.use_gpu}, Pose: {enable_pose})")
    
    def _load_models(self):
        """加载YOLO模型（PyTorch格式）"""
        # 获取项目根目录
        from config import ROOT_DIR
        
        # 加载人体检测模型
        print(f"[INFO] 正在加载人体检测模型...")
        person_model_path = ROOT_DIR / 'yolov8n.pt'
        self.person_model = YOLO(str(person_model_path))
        if self.use_gpu:
            self.person_model.to('cuda')
        print("[INFO] 人体检测模型加载完成")
        
        # 加载红领巾检测模型
        print(f"[INFO] 正在加载红领巾检测模型...")
        # 按照RedScarf_back的做法，使用yolov8n作为红领巾检测模型
        # 関键ideauff1ayolov8n是通用的一般目标检测模型（棄80个类别）
        # 我们通过其检测的所有物体作为“红领巾候选”，然后不用特定类捷
        redscarf_model_path = ROOT_DIR / 'yolov8n.pt'
        
        if not redscarf_model_path.exists():
            # 希望不会执行这一行
            print(f"[ERROR] 红领巾检测模型不存在: {redscarf_model_path}")
            raise FileNotFoundError(f"模型不存在: {redscarf_model_path}")
        
        self.redscarf_model = YOLO(str(redscarf_model_path))
        if self.use_gpu:
            self.redscarf_model.to('cuda')
        print(f"[INFO] 红领巾检测模型加载完成: {redscarf_model_path} ({redscarf_model_path.stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"[INFO] 作为报会: 使用预训练yolov8n模型検测所有物体，然后简化为红领巾候选，最后通过预位置并判定是否佩戴")
        
        # 加载姿态检测模型（可选）
        self.pose_detector = None
        self.salute_detector = None
        
        if self.enable_pose:
            try:
                print(f"[INFO] 正在加载姿态检测模型...")
                pose_model_path = ROOT_DIR / 'yolov8n-pose.pt'
                
                if not pose_model_path.exists():
                    print(f"[WARNING] 姿态模型不存在: {pose_model_path}，将禁用敬礼检测")
                    self.enable_pose = False
                else:
                    self.pose_detector = PoseDetector(
                        model_path=str(pose_model_path),
                        device=self.device,
                        conf_threshold=POSE_CONF_THRESHOLD
                    )
                    
                    # 初始化敬礼检测器
                    self.salute_detector = SaluteDetector(
                        angle_threshold=(SALUTE_ANGLE_MIN, SALUTE_ANGLE_MAX),
                        hand_head_distance_ratio=SALUTE_HAND_HEAD_RATIO,
                        strict_mode=SALUTE_STRICT_MODE
                    )
                    print("[INFO] 姿态检测模型加载完成")
            except Exception as e:
                print(f"[WARNING] 加载姿态模型失败: {e}，将禁用敬礼检测")
                self.enable_pose = False
                self.pose_detector = None
                self.salute_detector = None
    
    def detect_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        对单张图像进行检测（包含红领巾和敬礼姿态）
        采用RedScarf_back的流程：先检测红领巾，再检测人体并进行佩戴判定
        
        Args:
            image: 输入图像 (BGR格式)
        
        Returns:
            (result_image, info): 检测结果图像和统计信息
        """
        start_time = time.time()
        
        # 转换颜色空间用于检测
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = image.copy()
        
        # 1. 先检测红领巾 - 获取所有红领巾的位置
        # 使用较低的置信度阈值以确保捕获所有候选框
        redscarf_results = self.redscarf_model(image_rgb, verbose=False, conf=0.01)
        redscarf_boxes = []
        redscarf_candidates = []  # 记录所有候选用于调试
        
        # 调试: 打印所有检测结果
        print(f"[DEBUG] 红领巾模型检测结果数: {len(redscarf_results)}")
        
        for result_idx, result in enumerate(redscarf_results):
            # print(f"[DEBUG] 结果 {result_idx}: 检测框数 = {len(result.boxes)}")
            for box_idx, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                redscarf_candidates.append(conf)
                xyxy = box.xyxy[0].tolist()
                # print(f"[DEBUG] 框 {box_idx}: 置信度={conf:.4f}, 位置={[int(x) for x in xyxy]}")
                
                # 添加色彩过滤：检查框内是否有红色像素
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                roi = image_rgb[y1:y2, x1:x2]
                
                if roi.size > 0 and conf >= REDSCARF_CONF_THRESHOLD:
                    # 转换为HSV色彩空间
                    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                    
                    # 红色在HSV中的范围（考虑色调的循环性）
                    lower_red1 = np.array([0, 50, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 50, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
                    mask = cv2.bitwise_or(mask1, mask2)
                    
                    # 计算红色像素比例
                    red_pixel_ratio = np.sum(mask > 0) / mask.size
                    
                    # 只有红色像素比例足够高，才认定为红领巾
                    if red_pixel_ratio > 0.15:
                        redscarf_boxes.append(xyxy)
                        label = f'红领巾 {conf:.2f} ({red_pixel_ratio*100:.0f}%红)'
                        result_image = draw_detection_box(
                            result_image, xyxy, label,
                            COLOR_REDSCARF_BOX, BOX_LINE_THICKNESS, FONT_SCALE
                        )
        
        print(f"[DEBUG] 红领巾检测: 总候选={len(redscarf_candidates)}, 通过阈值={len(redscarf_boxes)}")
        
        # 2. 再检测人体 - 使用红领巾位置进行佩戴判定
        person_results = self.person_model(image_rgb, verbose=False)
        
        person_count = 0
        wearing_count = 0
        not_wearing_count = 0
        
        # 3. 检测姿态（如果启用）
        pose_detections = []
        salute_count = 0
        salute_results = []  # 存储每个人的敬礼检测结果
        
        if self.enable_pose and self.pose_detector:
            pose_detections = self.pose_detector.detect(image)
        
        for result in person_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # 仅处理人体类别 (class 0)
                if cls == 0 and conf >= PERSON_CONF_THRESHOLD:
                    person_count += 1
                    xyxy = box.xyxy[0].tolist()
                    
                    # 判断是否佩戴红领巾（参考RedScarf_back的逻辑）
                    is_wearing, matched_box = is_wearing_redscarf(
                        np.array(xyxy), redscarf_boxes,
                        iou_threshold=REDSCARF_IOU_THRESHOLD,
                        vertical_ratio_threshold=REDSCARF_VERTICAL_RATIO
                    )
                    
                    # 判断是否敬礼
                    salute_info = None
                    if self.enable_pose and self.salute_detector:
                        # 匹配当前人体框与姿态检测结果
                        matched_pose = self._match_person_to_pose(xyxy, pose_detections)
                        if matched_pose:
                            # 传递图像用于手掌检测
                            salute_info = self.salute_detector.detect_salute(
                                matched_pose['keypoints'], 
                                image=image  # 添加图像参数
                            )
                            if salute_info['is_saluting']:
                                salute_count += 1
                            salute_results.append(salute_info)
                    
                    # 确定标签和颜色（参考RedScarf_back的配色）
                    if is_wearing:
                        wearing_count += 1
                        base_color = COLOR_WEARING_REDSCARF  # 绿色
                        base_label = f'已佩戴红领巾 {conf:.2f}'
                    else:
                        not_wearing_count += 1
                        base_color = COLOR_NOT_WEARING  # 红色
                        base_label = f'未佩戴红领巾 {conf:.2f}'
                    
                    # 添加敬礼信息
                    if salute_info and salute_info['is_saluting']:
                        side_text = '左手' if salute_info['side'] == 'left' else '右手'
                        label = f"{base_label}\n{side_text}敬礼 {salute_info['score']:.0f}分"
                        # 如果敬礼姿态标准，使用紫色边框
                        if salute_info['score'] >= 85:
                            color = COLOR_SALUTING
                        else:
                            color = base_color
                    else:
                        label = base_label
                        color = base_color
                    
                    # 绘制人体检测框
                    result_image = draw_detection_box(
                        result_image, xyxy, label, 
                        color, BOX_LINE_THICKNESS, FONT_SCALE
                    )
        
        # 绘制姿态骨架（可选）
        if self.enable_pose and pose_detections:
            # 只绘制关键点和骨架，不绘制边界框（避免重复）
            result_image = self.pose_detector.draw_pose(
                result_image, pose_detections, 
                draw_bbox=False, draw_skeleton=True
            )
        
        # 计算FPS
        fps = 1.0 / (time.time() - start_time)
        
        # 绘制FPS
        if DISPLAY_FPS:
            result_image = draw_fps(result_image, fps)
        
        # 统计信息
        info = {
            "total_persons": person_count,
            "wearing_redscarf": wearing_count,
            "not_wearing": not_wearing_count,
            "redscarf_detected": len(redscarf_boxes),
            "redscarf_candidates": len(redscarf_candidates),  # 所有候选框数
            "redscarf_confidences": redscarf_candidates,  # 所有置信度
            "saluting": salute_count,
            "salute_results": salute_results,
            "fps": fps
        }
        
        return result_image, info
    
    def _match_person_to_pose(self, person_box: list, pose_detections: List[Dict]) -> Optional[Dict]:
        """
        匹配人体框与姿态检测结果
        
        Args:
            person_box: 人体边界框 [x1, y1, x2, y2]
            pose_detections: 姿态检测结果列表
        
        Returns:
            匹配的姿态检测结果或None
        """
        if not pose_detections:
            return None
        
        best_match = None
        best_iou = 0.0
        
        for pose_det in pose_detections:
            pose_bbox = pose_det['bbox']
            
            # 计算IoU
            from detector.utils import calculate_iou
            iou = calculate_iou(np.array(person_box), np.array(pose_bbox))
            
            # 选择IoU最大的匹配
            if iou > best_iou:
                best_iou = iou
                best_match = pose_det
        
        # 只返回IoU超过阈值的匹配
        if best_iou > 0.5:
            return best_match
        
        return None
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None):
        """
        对视频文件进行检测
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径 (可选)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 准备视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"[INFO] 开始处理视频: {video_path}")
        print(f"[INFO] 分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检测
                result_frame, info = self.detect_image(frame)
                
                # 写入输出视频
                if writer:
                    writer.write(result_frame)
                
                # 显示进度
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[INFO] 处理进度: {frame_count}/{total_frames} "
                          f"({100*frame_count/total_frames:.1f}%) - FPS: {info['fps']:.2f}")
                
                # 显示结果
                cv2.imshow("Red Scarf Detection", result_frame)
                
                # 按ESC退出
                if cv2.waitKey(1) == 27:
                    print("[INFO] 用户中断处理")
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"[INFO] 视频处理完成，共处理 {frame_count} 帧")
            if output_path:
                print(f"[INFO] 结果已保存至: {output_path}")
    
    def detect_camera(self, camera_id: int = 0):
        """
        使用摄像头进行实时检测
        
        Args:
            camera_id: 摄像头ID (默认为0)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        print(f"[INFO] 开始摄像头检测 (Camera ID: {camera_id})")
        print("[INFO] 按 ESC 键退出")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] 无法读取摄像头画面")
                    break
                
                # 检测
                result_frame, info = self.detect_image(frame)
                
                # 显示结果
                cv2.imshow("Red Scarf Detection - Camera", result_frame)
                
                # 按ESC退出
                if cv2.waitKey(1) == 27:
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] 摄像头检测已停止")
