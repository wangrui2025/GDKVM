import datetime

def log_message(log, message, to_console=True):
    """。
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    log.write(formatted_message + "\n")
    log.flush()
    if to_console:
        print(formatted_message)