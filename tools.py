import csv
import glob
import logging
import time
import os
from matplotlib import pyplot as plt

from modules.pretrain_options import *
from options import *


def format_dictionary(dictionary):
    # 获取所有的键和值
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # 获取最长的键和值的长度
    max_key_length = max(len(str(key)) for key in keys)
    max_value_length = max(len(str(value)) for value in values)

    # 打印表头
    header = f"| {'Key'.ljust(max_key_length)} | {'Value'.ljust(max_value_length)} |"
    separator = f"+{'-' * (max_key_length + 2)}+{'-' * (max_value_length + 2)}+"
    print(separator)
    print(header)
    print(separator)

    # 打印每一行数据
    for key, value in dictionary.items():
        row = f"| {str(key).ljust(max_key_length)} | {str(value).ljust(max_value_length)} |"
        print(row)

    print(separator)


def log_csv(opt):
    with open(os.path.join(opt['save_proj'], 'result.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in opt.items():
            writer.writerow([key, value])
        writer.writerow([' '])
        writer.writerow(['ID', 'Seq', 'SR', 'Mean_SR', 'Mean_PR', 'FPS'])


def train_log(info):
    current_time = time.strftime('%m-%d_%H:%M:%S', time.localtime())

    # 创建日志记录器
    logger = logging.getLogger('Mylogger')

    # 配置日志记录器
    logger.setLevel(logging.INFO)

    # 指定日志输出格式
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    num = len(glob.glob('./runs/train/*/'))
    if not os.path.exists('./runs/train/exp' + str(num)):
        os.mkdir('./runs/train/exp' + str(num))
    file_handler = logging.FileHandler('./runs/train/exp' + str(num) + '/' + str(current_time) + "-" + info + '.log')
    # 创建一个文件处理器，将日志输出到文件中
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)
    logger.info(info)
    logger.info(pretrain_opts)

    return logger


def plot_pic(pre, binloss, interloss, num):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    ax1.set_ylim(0, 1)
    ax1.set_title('pre')
    ax1.plot(pre[:200])

    ax2.set_ylim(0, 1)
    ax2.set_title('binloss')
    ax2.plot(binloss[:200])

    ax3.set_ylim(0, 5)
    ax3.set_title('interloss')
    ax3.plot(interloss[:200])

    plt.tight_layout()
    plt.savefig('./runs/train/exp' + str(num) + '/' + 'train_curve.jpg')
    plt.close()
