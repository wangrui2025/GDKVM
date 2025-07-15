import datetime

def log_message(log, message, to_console=True):
    """
    将日志写入文件，同时选择性地打印到控制台。
    
    Args:
        log: 文件句柄，用于写入日志。
        message: 字符串，写入的日志内容。
        to_console: 布尔值，是否将日志同时打印到控制台。
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    log.write(formatted_message + "\n")
    log.flush()
    if to_console:
        print(formatted_message)