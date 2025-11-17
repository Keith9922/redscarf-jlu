"""
红领巾检测系统 - Gradio Web界面
基于Gradio框架构建的交互式Web应用
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from detection_service import RedScarfDetectionService
from config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE


class GradioApp:
    """Gradio Web应用"""
    
    def __init__(self):
        """初始化应用"""
        print("[INFO] 正在初始化红领巾检测系统...")
        self.detector = RedScarfDetectionService()
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
- **处理速度**: {info['fps']:.2f} FPS
- **佩戴率**: {(info['wearing_redscarf']/info['total_persons']*100 if info['total_persons'] > 0 else 0):.1f}%

---
**说明**: 
- 🟢 绿色框 = 已佩戴红领巾
- 🔴 红色框 = 未佩戴红领巾  
- 🔵 青色框 = 红领巾位置
"""
        
        return result_image_rgb, info_text
    
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
                
                ### 基于YOLOv8 + OpenVINO的智能识别系统
                
                本系统可以自动识别小学生是否正确佩戴红领巾，帮助学校进行规范化管理。
                """,
                elem_classes="title"
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
                    - ✅ 实时处理反馈
                    - ✅ 可视化结果展示
                    - ✅ 统计信息输出
                    
                    ### 技术架构
                    - **目标检测**: YOLOv8
                    - **推理加速**: OpenVINO
                    - **Web框架**: Gradio
                    - **图像处理**: OpenCV
                    
                    ### 检测逻辑
                    系统采用两阶段检测策略:
                    1. **第一阶段**: 使用YOLOv8模型检测图像中的所有人体
                    2. **第二阶段**: 使用专门训练的模型检测红领巾位置
                    3. **判断逻辑**: 通过IoU和位置关系判断每个人是否佩戴红领巾
                    
                    ### 判断标准
                    - 红领巾必须出现在人体框的上半部分(颈部/胸部区域)
                    - 红领巾框与人体框有足够的重叠度(IoU)
                    - 综合位置得分和置信度进行判断
                    
                    ### 使用场景
                    - 学校日常检查
                    - 活动监督
                    - 统计分析
                    - 自动化管理
                    
                    ### 开发信息
                    - **版本**: v2.0
                    - **更新日期**: 2024
                    - **开发者**: Vicwxy Wangxinyu & AI Assistant
                    
                    ---
                    
                    💡 **提示**: 为获得最佳检测效果，建议上传清晰、光线充足的图片
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
