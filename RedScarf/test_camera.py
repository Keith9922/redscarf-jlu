#!/usr/bin/env python3
"""
摄像头功能测试脚本
测试红领巾检测系统的实时摄像头功能
"""
import cv2
import sys
from pathlib import Path

# 添加项目目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from detection_service import RedScarfDetectionService


def test_camera(camera_id: int = 0, test_frames: int = 10):
    """
    测试摄像头检测功能
    
    Args:
        camera_id: 摄像头ID
        test_frames: 测试帧数
    """
    print(f"[INFO] 测试摄像头 {camera_id}...")
    
    # 初始化检测服务
    detector = RedScarfDetectionService()
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}")
        return False
    
    print(f"[INFO] 摄像头已打开，将采集 {test_frames} 帧进行测试...")
    
    frame_count = 0
    total_persons = 0
    total_wearing = 0
    total_saluting = 0
    
    try:
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 无法读取摄像头画面")
                break
            
            # 执行检测
            result_frame, info = detector.detect_image(frame)
            
            frame_count += 1
            total_persons += info['total_persons']
            total_wearing += info['wearing_redscarf']
            total_saluting += info['saluting']
            
            print(f"[Frame {frame_count}] 人数: {info['total_persons']}, "
                  f"佩戴红领巾: {info['wearing_redscarf']}, "
                  f"敬礼: {info['saluting']}, "
                  f"FPS: {info['fps']:.2f}")
            
            # 显示结果（可选）
            cv2.imshow("Camera Test", result_frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # 输出统计
    print("\n" + "="*60)
    print("测试完成统计:")
    print(f"- 总帧数: {frame_count}")
    print(f"- 检测到的总人数: {total_persons}")
    print(f"- 佩戴红领巾总数: {total_wearing}")
    print(f"- 敬礼总次数: {total_saluting}")
    if frame_count > 0:
        print(f"- 平均人数/帧: {total_persons/frame_count:.2f}")
        print(f"- 平均佩戴率: {(total_wearing/max(total_persons, 1)*100):.1f}%")
    print("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="摄像头检测功能测试")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID (默认: 0)")
    parser.add_argument("--frames", type=int, default=10, help="测试帧数 (默认: 10)")
    
    args = parser.parse_args()
    
    success = test_camera(camera_id=args.camera, test_frames=args.frames)
    
    sys.exit(0 if success else 1)
