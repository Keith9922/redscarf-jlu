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
from config import (
    PERSON_MODEL_PATH, REDSCARF_MODEL_PATH, DEVICE,
    PERSON_CONF_THRESHOLD, REDSCARF_CONF_THRESHOLD,
    COLOR_WEARING_REDSCARF, COLOR_NOT_WEARING, COLOR_REDSCARF_BOX,
    BOX_LINE_THICKNESS, FONT_SCALE, DISPLAY_FPS,
    REDSCARF_IOU_THRESHOLD, REDSCARF_VERTICAL_RATIO
)


class RedScarfDetectionService:
    """红领巾检测服务"""
    
    def __init__(self, device: str = DEVICE):
        """
        初始化检测服务
        
        Args:
            device: 运行设备 ("CPU" 或 "GPU")
        """
        self.device = device
        self.use_gpu = device.upper() == 'GPU' and torch.cuda.is_available()
        
        # 加载模型
        self._load_models()
        
        print(f"[INFO] 红领巾检测服务初始化完成 (Device: {device}, GPU: {self.use_gpu})")
    
    def _load_models(self):
        """加载YOLO模型（PyTorch格式）"""
        # 加载人体检测模型
        print(f"[INFO] 正在加载人体检测模型...")
        self.person_model = YOLO('yolov8n.pt')
        if self.use_gpu:
            self.person_model.to('cuda')
        print("[INFO] 人体检测模型加载完成")
        
        # 加载红领巾检测模型
        print(f"[INFO] 正在加载红领巾检测模型...")
        self.redscarf_model = YOLO('yolov8n.pt')  # 使用通用模型，后续可替换为专用权重
        if self.use_gpu:
            self.redscarf_model.to('cuda')
        print("[INFO] 红领巾检测模型加载完成")
    
    def detect_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        对单张图像进行检测
        
        Args:
            image: 输入图像 (BGR格式)
        
        Returns:
            (result_image, info): 检测结果图像和统计信息
        """
        start_time = time.time()
        
        # 转换颜色空间用于检测
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = image.copy()
        
        # 1. 检测红领巾 - 使用YOLO模型直接推理
        redscarf_results = self.redscarf_model(image_rgb, verbose=False)
        redscarf_boxes = []
        
        for result in redscarf_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= REDSCARF_CONF_THRESHOLD:
                    xyxy = box.xyxy[0].tolist()
                    redscarf_boxes.append(xyxy)
                    # 绘制红领巾检测框
                    label = f'红领巾 {conf:.2f}'
                    result_image = draw_detection_box(
                        result_image, xyxy, label, 
                        COLOR_REDSCARF_BOX, BOX_LINE_THICKNESS, FONT_SCALE
                    )
        
        # 2. 检测人体 - 使用YOLO模型直接推理
        person_results = self.person_model(image_rgb, verbose=False)
        
        person_count = 0
        wearing_count = 0
        not_wearing_count = 0
        
        for result in person_results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # 仅处理人体类别 (class 0)
                if cls == 0 and conf >= PERSON_CONF_THRESHOLD:
                    person_count += 1
                    xyxy = box.xyxy[0].tolist()
                    
                    # 判断是否佩戴红领巾
                    is_wearing, matched_box = is_wearing_redscarf(
                        np.array(xyxy), redscarf_boxes,
                        iou_threshold=REDSCARF_IOU_THRESHOLD,
                        vertical_ratio_threshold=REDSCARF_VERTICAL_RATIO
                    )
                    
                    if is_wearing:
                        wearing_count += 1
                        color = COLOR_WEARING_REDSCARF
                        label = f'已佩戴红领巾 {conf:.2f}'
                    else:
                        not_wearing_count += 1
                        color = COLOR_NOT_WEARING
                        label = f'未佩戴红领巾 {conf:.2f}'
                    
                    # 绘制人体检测框
                    result_image = draw_detection_box(
                        result_image, xyxy, label, 
                        color, BOX_LINE_THICKNESS, FONT_SCALE
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
            "fps": fps
        }
        
        return result_image, info
    
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
