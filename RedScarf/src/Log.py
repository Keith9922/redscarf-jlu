"""日志模块 - 线程安全的日志记录"""
import datetime
import logging
from logging.handlers import RotatingFileHandler
import threading

lock = threading.Lock()

def log(message: str, file_path: str = "./logs/log.log", level: str = 'INFO', 
        time_format: str = '%Y-%m-%d %H:%M:%S', max_size: int = 1024*1024, 
        backup_count: int = 5) -> None:
    """记录日志信息到文件和控制台
    
    Args:
        message: 日志消息
        file_path: 日志文件路径
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        time_format: 时间格式字符串
        max_size: 日志文件最大大小 (字节)
        backup_count: 备份日志文件数量
    """
    datetime_object = datetime.datetime.now()
    datetime_str = datetime_object.strftime(time_format)
    handler = RotatingFileHandler(file_path, maxBytes=max_size, backupCount=backup_count)
    formatter = logging.Formatter(f'[{datetime_str} %(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    
    # 设置日志级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'INFO': logging.INFO
    }
    logger.setLevel(level_map.get(level, logging.INFO))
    
    # 线程安全地输出日志
    with lock:
        if level == 'DEBUG':
            logger.debug(message)
        elif level == 'WARNING':
            logger.warning(message)
        elif level == 'ERROR':
            logger.error(message)
        else:
            logger.info(message)
        print(f'[{datetime_str} {level}] {message}')