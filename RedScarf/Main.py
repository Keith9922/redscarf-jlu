#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
红领巾检测系统 - 摄像头实时检测模式
使用OpenCV和OpenVINO进行实时检测

作者: Vicwxy Wangxinyu
更新: 2024
"""

from detection_service import RedScarfDetectionService
from Log import log
import sys
import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='红领巾检测系统 - 实时检测',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='摄像头ID (默认: 0)'
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='检测单张图片'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='检测视频文件'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出视频路径 (仅视频模式)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    log("Red Scarf Detection System Starting (by Vicwxy Wangxinyu)")
    
    # 解析参数
    args = parse_args()
    
    try:
        # 初始化检测服务
        log("Initializing detection service...")
        detector = RedScarfDetectionService()
        log("Detection service initialized successfully")
        
        # 根据参数选择模式
        if args.image:
            # 图片模式
            log(f"Image detection mode: {args.image}")
            import cv2
            image = cv2.imread(args.image)
            if image is None:
                log(f"ERROR: Cannot read image: {args.image}", level='ERROR')
                return 1
            
            result, info = detector.detect_image(image)
            
            log(f"Detection complete: {info}")
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif args.video:
            # 视频模式
            log(f"Video detection mode: {args.video}")
            detector.detect_video(args.video, args.output)
            
        else:
            # 摄像头模式 (默认)
            log(f"Camera detection mode (Camera ID: {args.camera})")
            log("Press ESC to exit...")
            detector.detect_camera(args.camera)
        
        log("Program finished successfully")
        return 0
        
    except KeyboardInterrupt:
        log("Program interrupted by user", level='WARNING')
        return 0
    
    except Exception as e:
        log(f"ERROR: {str(e)}", level='ERROR')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
