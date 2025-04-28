import os
import time
import datetime
import shutil


def make_empty_dir(new_dir):
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)


def get_timestamp():
    return str(time.time()).replace('.', '')


def get_formatted_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ​​获取当前时间并格式化为字符串​​，通常用于日志记录、文件命名等场景。
