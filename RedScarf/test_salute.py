"""
敬礼检测功能测试脚本
测试YOLOv8-Pose姿态检测和敬礼识别算法
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from detector.pose_detector import PoseDetector
from detector.salute_detector import SaluteDetector


def test_pose_detection():
    """测试姿态检测功能"""
    print("="*60)
    print("测试姿态检测功能")
    print("="*60)
    
    # 获取项目根目录
    root_dir = Path(__file__).parent
    
    # 模型路径
    pose_model_path = root_dir / 'yolov8n-pose.pt'
    
    if not pose_model_path.exists():
        print(f"[ERROR] 姿态模型不存在: {pose_model_path}")
        print("请先下载 yolov8n-pose.pt 模型文件")
        return False
    
    # 初始化检测器
    print("\n[INFO] 初始化姿态检测器...")
    pose_detector = PoseDetector(
        model_path=str(pose_model_path),
        device='CPU',
        conf_threshold=0.5
    )
    
    # 初始化敬礼检测器
    print("[INFO] 初始化敬礼检测器...")
    salute_detector = SaluteDetector(
        angle_threshold=(60, 120),
        hand_head_distance_ratio=0.3,
        strict_mode=False
    )
    
    # 测试图像路径
    test_images = [
        root_dir / 'data' / 'images' / 'test.jpg',
        root_dir / '14_debug.jpg',
        root_dir / '15_debug.jpg'
    ]
    
    # 选择存在的测试图像
    test_image = None
    for img_path in test_images:
        if img_path.exists():
            test_image = img_path
            break
    
    if not test_image:
        print("[WARNING] 没有找到测试图像，使用摄像头测试")
        test_camera(pose_detector, salute_detector)
        return True
    
    print(f"\n[INFO] 测试图像: {test_image}")
    
    # 读取图像
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"[ERROR] 无法读取图像: {test_image}")
        return False
    
    print(f"[INFO] 图像尺寸: {image.shape}")
    
    # 姿态检测
    print("\n[INFO] 开始姿态检测...")
    pose_detections = pose_detector.detect(image)
    
    print(f"[INFO] 检测到 {len(pose_detections)} 个人体姿态")
    
    # 绘制姿态
    result_image = pose_detector.draw_pose(
        image.copy(),
        pose_detections,
        draw_bbox=True,
        draw_skeleton=True
    )
    
    # 敬礼检测
    print("\n[INFO] 敬礼检测结果:")
    for i, detection in enumerate(pose_detections, 1):
        keypoints = detection['keypoints']
        salute_result = salute_detector.detect_salute(keypoints)
        
        print(f"\n人员 {i}:")
        print(f"  - 是否敬礼: {salute_result['is_saluting']}")
        print(f"  - 敬礼手侧: {salute_result['side']}")
        print(f"  - 姿态得分: {salute_result['score']:.1f}/100")
        print(f"  - 手肘角度: {salute_result['details']['elbow_angle']:.1f}°")
        print(f"  - 手部位置: {salute_result['details']['hand_position']}")
        print(f"  - 手部高度: {salute_result['details']['hand_height']}")
        print(f"  - 整体评价: {salute_result['details']['posture']}")
        
        # 在图像上添加敬礼信息
        if salute_result['is_saluting']:
            bbox = detection['bbox']
            x1, y1 = int(bbox[0]), int(bbox[1])
            side_text = '左手' if salute_result['side'] == 'left' else '右手'
            text = f"{side_text}敬礼 {salute_result['score']:.0f}分"
            
            cv2.putText(result_image, text, (x1, y1 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    # 显示结果
    print("\n[INFO] 显示检测结果（按任意键退出）")
    
    # 调整显示尺寸
    display_height = 800
    scale = display_height / result_image.shape[0]
    display_width = int(result_image.shape[1] * scale)
    display_image = cv2.resize(result_image, (display_width, display_height))
    
    cv2.imshow("Pose Detection & Salute Recognition", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    output_path = root_dir / 'pose_detection_result.jpg'
    cv2.imwrite(str(output_path), result_image)
    print(f"\n[INFO] 结果已保存至: {output_path}")
    
    return True


def test_camera(pose_detector, salute_detector):
    """使用摄像头测试"""
    print("\n[INFO] 启动摄像头测试...")
    print("[INFO] 按 ESC 键退出")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取摄像头画面")
            break
        
        # 姿态检测
        pose_detections = pose_detector.detect(frame)
        
        # 绘制姿态
        result_frame = pose_detector.draw_pose(
            frame.copy(),
            pose_detections,
            draw_bbox=True,
            draw_skeleton=True
        )
        
        # 敬礼检测
        for detection in pose_detections:
            keypoints = detection['keypoints']
            salute_result = salute_detector.detect_salute(keypoints)
            
            if salute_result['is_saluting']:
                bbox = detection['bbox']
                x1, y1 = int(bbox[0]), int(bbox[1])
                side_text = '左手' if salute_result['side'] == 'left' else '右手'
                text = f"{side_text}敬礼 {salute_result['score']:.0f}分"
                
                cv2.putText(result_frame, text, (x1, y1 - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # 显示结果
        cv2.imshow("Camera - Pose & Salute Detection", result_frame)
        
        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """主函数"""
    try:
        success = test_pose_detection()
        
        if success:
            print("\n" + "="*60)
            print("✅ 测试完成！")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 测试失败")
            print("="*60)
    
    except Exception as e:
        print(f"\n[ERROR] 测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
