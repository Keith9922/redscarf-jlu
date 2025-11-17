"""
红领巾检测系统配置文件
"""
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 模型配置
PERSON_MODEL_NAME = "yolov8n"
REDSCARF_MODEL_NAME = "redscarf"
POSE_MODEL_NAME = "yolov8n-pose"  # YOLOv8-Pose模型
DEVICE = "CPU"  # 可选: CPU, GPU

# 模型路径
PERSON_MODEL_PATH = ROOT_DIR / f"models/{PERSON_MODEL_NAME}_openvino_model/{PERSON_MODEL_NAME}.xml"
REDSCARF_MODEL_PATH = ROOT_DIR / f"models/{REDSCARF_MODEL_NAME}_openvino_model/{REDSCARF_MODEL_NAME}.xml"
POSE_MODEL_PATH = ROOT_DIR / "yolov8n-pose.pt"  # YOLOv8-Pose模型路径

# 检测参数
PERSON_CONF_THRESHOLD = 0.5      # 人体检测置信度阈值
REDSCARF_CONF_THRESHOLD = 0.45    # 红领巾检测置信度阈值 (混合色彩过滤后)
POSE_CONF_THRESHOLD = 0.5        # 姿态检测置信度阈值
NMS_IOU_THRESHOLD = 0.7          # NMS IOU阈值
MAX_DETECTIONS = 300             # 最大检测数量

# 红领巾佩戴判断参数
REDSCARF_IOU_THRESHOLD = 0.05    # 红领巾与人体框IoU阈值 (允许灵活匹配)
REDSCARF_VERTICAL_RATIO = 0.5    # 红领巾在人体框中的垂直位置比例

# 敬礼检测参数
SALUTE_ANGLE_MIN = 60            # 敬礼手肘最小角度
SALUTE_ANGLE_MAX = 120           # 敬礼手肘最大角度
SALUTE_HAND_HEAD_RATIO = 0.3    # 手部到头部距离比例
SALUTE_STRICT_MODE = False       # 是否使用严格模式

# 日志配置
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "log.log"
LOG_LEVEL = "INFO"
LOG_MAX_SIZE = 1024 * 1024      # 1MB
LOG_BACKUP_COUNT = 5

# Gradio界面配置
GRADIO_SERVER_NAME = "0.0.0.0"   # 服务器地址
GRADIO_SERVER_PORT = 7860        # 服务器端口
GRADIO_SHARE = False             # 是否创建公共链接

# 显示配置
DISPLAY_FPS = True               # 是否显示FPS
BOX_LINE_THICKNESS = 2           # 边框线条粗细
FONT_SCALE = 0.6                 # 字体大小

# 颜色配置 (BGR格式)
COLOR_WEARING_REDSCARF = (0, 255, 0)    # 佩戴红领巾: 绿色
COLOR_NOT_WEARING = (0, 0, 255)         # 未佩戴: 红色
COLOR_REDSCARF_BOX = (255, 255, 0)      # 红领巾框: 青色
COLOR_SALUTING = (255, 0, 255)          # 敬礼姿态: 紫色
COLOR_NOT_SALUTING = (128, 128, 128)    # 未敬礼: 灰色
COLOR_TEXT = (255, 255, 255)            # 文字颜色: 白色
