"""
红领巾检测系统 - Gradio Web界面
基于Gradio框架构建的交互式Web应用
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import threading
import time
from typing import Tuple, Dict, Optional
import random
import os
import sys

# macOS OpenCV摄像头权限处理
if sys.platform == 'darwin':
    os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

from detection_service import RedScarfDetectionService
from config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE


class GradioApp:
    """Gradio Web应用"""
    
    def __init__(self):
        """初始化应用"""
        print("[INFO] 正在初始化红领巾检测系统...")
        self.detector = RedScarfDetectionService()
        self.camera_running = False
        self.latest_frame = None
        self.latest_info = None
        self.praise_message = ""
        self.last_praise_time = 0
        print("[INFO] 系统初始化完成!")
    
    def detect_image_interface(self, image: np.ndarray):
        """
        图像检测接口
        
        Args:
            image: PIL Image或numpy array
        
        Returns:
            (result_image, info_text): 检测结果图像和信息文本
        """
        if image is None:
            return None, "请上传图片"
        
        # 确保图像是numpy array格式
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 确保是BGR格式
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 执行检测
        result_image, info = self.detector.detect_image(image)
        
        # 转换回RGB用于显示
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # 生成信息文本
        info_text = f"""
### 检测结果

- **检测到的人数**: {info['total_persons']} 人
- **已佩戴红领巾**: {info['wearing_redscarf']} 人 ✅
- **未佩戴红领巾**: {info['not_wearing']} 人 ❌
- **检测到的红领巾**: {info['redscarf_detected']} 个
- **红领巾候选框**: {info.get('redscarf_candidates', 0)} 个
- **正在敬礼**: {info.get('saluting', 0)} 人 👋
- **处理速度**: {info['fps']:.2f} FPS
- **佩戴率**: {(info['wearing_redscarf']/info['total_persons']*100 if info['total_persons'] > 0 else 0):.1f}%
"""
        
        # 添加红领巾置信度信息
        if info.get('redscarf_confidences'):
            confs = info['redscarf_confidences']
            if confs:
                max_conf = max(confs)
                info_text += f"\n### 调试信息\n"
                info_text += f"- **最高红领巾置信度**: {max_conf:.3f}\n"
                if len(confs) > 1:
                    info_text += f"- **平均红领巾置信度**: {sum(confs)/len(confs):.3f}\n"
                info_text += f"- **当前阈值**: 0.3\n"
                info_text += f"- **提示**: 如果上述置信度 > 0.3 但未被检测到，请检查模型或图像\n"
        
        # 添加敬礼详细信息
        if info.get('salute_results'):
            info_text += "\n### 敬礼姿态详情\n\n"
            for i, salute_result in enumerate(info.get('salute_results', []), 1):
                if salute_result['is_saluting']:
                    side_text = '左手' if salute_result['side'] == 'left' else '右手'
                    score = salute_result['score']
                    details = salute_result['details']
                    
                    info_text += f"**人员 {i}**: {side_text}敬礼 (得分: {score:.1f}/100)\n"
                    info_text += f"- 手肘角度: {details['elbow_angle']:.1f}°\n"
                    info_text += f"- 手部位置: {details['hand_position']}\n"
                    info_text += f"- 手部高度: {details['hand_height']}\n"
                    info_text += f"- 整体评价: {details['posture']}\n\n"
        
        info_text += """
---
**说明**: 
- 🟢 绿色框 = 已佩戴红领巾
- 🔴 红色框 = 未佩戴红领巾  
- 🔵 青色框 = 红领巾位置
- 🟣 紫色框 = 标准敬礼姿态
- 🟡 黄色骨架 = 人体姿态关键点
"""
        
        return result_image_rgb, info_text
    
    def _generate_praise(self) -> str:
        """生成鼓励信息"""
        praise_list = [
            "🌟 太棒了！正确佩戴红领巾！",
            "🎉 优秀！标准敬礼姿态！",
            "⭐ 你是好少年！",
            "👍 敬礼姿态标准，继续加油！",
            "🏆 完美的敬礼！",
            "✨ 红领巾佩戴得很好！",
            "💪 继续保持这样的好习惯！",
            "🎓 这就是少先队员的风采！",
            "👏 敬礼动作棒棒哒！",
            "🌈 展现红领巾的光彩！",
        ]
        return random.choice(praise_list)
    
    def camera_detection_interface(self) -> Tuple[Optional[np.ndarray], str]:
        """
        摄像头实时检测接口
        返回当前帧和信息
        """
        if self.latest_frame is not None:
            # 将最新帧转换为RGB用于显示
            if len(self.latest_frame.shape) == 3 and self.latest_frame.shape[2] == 3:
                result_image_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            else:
                result_image_rgb = self.latest_frame
            
            # 生成信息文本
            if self.latest_info:
                info = self.latest_info
                info_text = f"""
### 实时检测结果

- **检测到的人数**: {info['total_persons']} 人
- **已佩戴红领巾**: {info['wearing_redscarf']} 人 ✅
- **未佩戴红领巾**: {info['not_wearing']} 人 ❌
- **正在敬礼**: {info.get('saluting', 0)} 人 👋
- **检测速度**: {info['fps']:.2f} FPS
- **佩戴率**: {(info['wearing_redscarf']/max(info['total_persons'], 1)*100):.1f}%
"""
                
                # 如果检测到正确佩戴红领巾且敬礼，添加鼓励信息
                if info['wearing_redscarf'] > 0 and info.get('saluting', 0) > 0:
                    current_time = time.time()
                    if not self.praise_message or (current_time - self.last_praise_time) > 3:
                        self.praise_message = self._generate_praise()
                        self.last_praise_time = current_time
                    info_text += f"\n---\n### 🎉 鼓励信息\n\n{self.praise_message}"
                else:
                    self.praise_message = ""
                
                return result_image_rgb, info_text
        
        # 如果摄像头正在启动中
        if self.camera_running:
            return None, "⏳ 摄像头启动中，请稍候...\n\nmacOS用户：\n- 首次使用需要在系统偏好设置中授予摄像头权限\n- 如果仍然无法工作，请检查是否有其他应用占用摄像头\n- 尝试更改摄像头ID（如改为1）"
        
        return None, "等待摄像头输入..."
    
    def _camera_thread(self, camera_id: int = 0):
        """摄像头检测线程"""
        try:
            print(f"[INFO] 摄像头线程启动，开始初始化摄像头 {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            
            # 增加初始化超时
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print(f"[ERROR] 无法打开摄像头: {camera_id}")
                print(f"[HELP] macOS用户请确保：")
                print(f"      1. 已在系统偏好设置中授予摄像头权限")
                print(f"      2. 没有其他应用占用摄像头")
                print(f"      3. 尝试更改摄像头ID（如改为1）")
                self.camera_running = False
                return
            
            print(f"[INFO] 摄像头已启动 (ID: {camera_id})")
            
            frame_count = 0
            try:
                while self.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[WARNING] 无法读取摄像头帧")
                        break
                    
                    # 检测
                    result_frame, info = self.detector.detect_image(frame)
                    
                    # 更新最新帧和信息
                    self.latest_frame = result_frame
                    self.latest_info = info
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[INFO] 摄像头运行中... 已处理 {frame_count} 帧")
                    
                    # 为了避免过度占用CPU，适度延迟
                    time.sleep(0.01)
            
            except Exception as e:
                print(f"[ERROR] 摄像头检测出错: {e}")
            
            finally:
                cap.release()
                self.camera_running = False
                print(f"[INFO] 摄像头已关闭，共处理 {frame_count} 帧")
        
        except Exception as e:
            print(f"[ERROR] 摄像头线程异常: {e}")
            self.camera_running = False
    
    def start_camera(self, camera_id: int = 0) -> str:
        """启动摄像头"""
        if not self.camera_running:
            self.camera_running = True
            self.latest_frame = None
            self.latest_info = None
            print(f"\n[INFO] 正在启动摄像头 {int(camera_id)}...")
            thread = threading.Thread(target=self._camera_thread, args=(int(camera_id),), daemon=True)
            thread.start()
            print(f"[INFO] 摄像头启动线程已创建")
            return "⏳ 摄像头启动中，请稍候..."
        return "⚠️ 摄像头已在运行中"
    
    def stop_camera(self) -> str:
        """停止摄像头"""
        self.camera_running = False
        time.sleep(0.5)  # 等待线程关闭
        return "✅ 摄像头已停止"
    
    def create_interface(self):
        """创建Gradio界面"""
        
        # 自定义CSS样式
        custom_css = """
        .gradio-container {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .description {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        """
        
        # 创建界面
        with gr.Blocks(css=custom_css, title="红领巾检测系统") as app:
            
            gr.Markdown(
                """
                # 🎓 小学生红领巾佩戴检测系统
                
                ### 基于YOLOv8 + YOLOv8-Pose + OpenVINO的智能识别系统
                
                本系统可以自动识别小学生是否正确佩戴红领巾，并检测敬礼姿态是否标准。
                """,
                elem_classes="title"
            )
            
            with gr.Tab("🎥 摄像头实时检测"):
                gr.Markdown("### 实时检测摄像头画面")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        camera_output = gr.Image(
                            label="摄像头画面",
                            type="numpy",
                            height=400
                        )
                        
                        camera_info = gr.Markdown(
                            label="检测信息",
                            value="等待启动..."
                        )
                    
                    with gr.Column(scale=1):
                        with gr.Row():
                            start_btn = gr.Button(
                                "▶️ 启动摄像头",
                                variant="primary",
                                size="lg"
                            )
                            stop_btn = gr.Button(
                                "⏹️ 停止摄像头",
                                variant="stop",
                                size="lg"
                            )
                        
                        camera_id_input = gr.Slider(
                            label="摄像头ID",
                            minimum=0,
                            maximum=5,
                            value=0,
                            step=1
                        )
                        
                        status_text = gr.Textbox(
                            label="状态",
                            value="就绪",
                            interactive=False
                        )
                        
                        gr.Markdown(
                            """
                            **使用说明**:
                            1. 设置摄像头ID（通常为0）
                            2. 点击"启动摄像头"开始实时检测
                            3. 系统会自动检测红领巾佩戴和敬礼姿态
                            4. 点击"停止摄像头"结束检测
                            
                            **检测结果说明**:
                            - 🟢 绿色框 = 已佩戴红领巾
                            - 🔴 红色框 = 未佩戴红领巾
                            - 🟣 紫色框 = 标准敬礼姿态
                            - 🟡 骨架线 = 人体关键点
                            
                            **鼓励机制**:
                            当检测到用户正确佩戴红领巾且做出敬礼动作时，系统会给出鼓励提示！
                            """
                        )
                
                # 定时更新函数
                def update_camera():
                    result, info = self.camera_detection_interface()
                    return result, info
                
                # 使用Timer组件持续更新（每100ms）
                timer = gr.Timer(value=0.1)
                timer.tick(
                    fn=update_camera,
                    outputs=[camera_output, camera_info]
                )
                
                # 绑定按钮事件
                start_btn.click(
                    fn=self.start_camera,
                    inputs=[camera_id_input],
                    outputs=[status_text]
                )
                
                stop_btn.click(
                    fn=self.stop_camera,
                    outputs=[status_text]
                )
            
            
            with gr.Tab("📷 图片检测"):
                gr.Markdown("### 上传图片进行检测")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="上传图片",
                            type="numpy",
                            height=400
                        )
                        
                        detect_btn = gr.Button(
                            "🔍 开始检测",
                            variant="primary",
                            size="lg"
                        )
                        
                        gr.Markdown(
                            """
                            **使用说明**:
                            1. 点击上传图片或拖拽图片到框内
                            2. 点击"开始检测"按钮
                            3. 查看右侧检测结果
                            
                            **支持格式**: JPG, PNG, BMP等常见图片格式
                            """
                        )
                    
                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="检测结果",
                            type="numpy",
                            height=400
                        )
                        
                        info_output = gr.Markdown(
                            label="检测信息",
                            value="等待上传图片..."
                        )
                
                # 绑定事件
                detect_btn.click(
                    fn=self.detect_image_interface,
                    inputs=[image_input],
                    outputs=[image_output, info_output]
                )
                
                # 示例图片
                gr.Examples(
                    examples=[
                        str(p) for p in Path("data/images").glob("*.jpg")
                        if Path("data/images").exists()
                    ][:5],  # 最多显示5个示例
                    inputs=image_input,
                    label="示例图片"
                )
            
            with gr.Tab("ℹ️ 系统信息"):
                gr.Markdown(
                    """
                    ## 系统介绍
                    
                    ### 功能特点
                    - ✅ 高精度人体检测
                    - ✅ 红领巾佩戴识别
                    - ✅ 人体姿态检测
                    - ✅ 敬礼动作识别
                    - ✅ 实时处理反馈
                    - ✅ 可视化结果展示
                    - ✅ 统计信息输出
                    - ✅ 摄像头实时检测
                    - ✅ 智能鼓励提示
                    
                    ### 技术架构
                    - **目标检测**: YOLOv8
                    - **姿态识别**: YOLOv8-Pose
                    - **推理加速**: OpenVINO (可选)
                    - **Web框架**: Gradio
                    - **图像处理**: OpenCV
                    
                    ### 检测逻辑
                    系统采用多模型协同检测策略:
                    1. **第一阶段**: 使用YOLOv8模型检测图像中的所有人体
                    2. **第二阶段**: 使用专门训练的模型检测红领巾位置
                    3. **第三阶段**: 使用YOLOv8-Pose检测人体关键点
                    4. **第四阶段**: 基于关键点算法判断敬礼姿态
                    5. **判断逻辑**: 通过IoU和位置关系判断每个人是否佩戴红领巾
                    
                    ### 敬礼判断标准
                    - 手肘角度在 60°-120° 范围内
                    - 手部位置在头部附近
                    - 手肘抬起高于肩膀
                    - 综合得分超过60分（标准85+分）
                    
                    ### 摄像头检测说明
                    - 支持多个摄像头输入（通过摄像头ID选择）
                    - 实时处理视频流，每帧进行目标检测和姿态识别
                    - 当检测到正确佩戴红领巾且敬礼时，自动生成随机鼓励信息
                    - 鼓励信息每3秒更新一次，防止重复
                    
                    ### 使用场景
                    - 学校日常检查
                    - 活动监督
                    - 敬礼动作训练
                    - 统计分析
                    - 自动化管理
                    - 教室/集会实时监控
                    
                    ### 开发信息
                    - **版本**: v4.0 (新增摄像头实时检测+鼓励提示)
                    - **更新日期**: 2024
                    - **开发者**: Vicwxy Wangxinyu & AI Assistant
                    
                    ---
                    
                    💡 **提示**: 为获得最佳检测效果，建议在光线充足的环境中使用，确保摄像头清晰，人物姿态完整可见。
                    """
                )
        
        return app
    
    def launch(self):
        """启动应用"""
        app = self.create_interface()
        
        print("\n" + "="*60)
        print("🚀 红领巾检测系统启动中...")
        print("="*60)
        
        app.launch(
            server_name=GRADIO_SERVER_NAME,
            server_port=GRADIO_SERVER_PORT,
            share=GRADIO_SHARE,
            show_error=True
        )


def main():
    """主函数"""
    # 确保在正确的目录中
    import os
    from pathlib import Path
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"[INFO] 工作目录: {os.getcwd()}")
    
    try:
        app = GradioApp()
        app.launch()
    except KeyboardInterrupt:
        print("\n[INFO] 系统已关闭")
    except Exception as e:
        print(f"\n[ERROR] 系统错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
