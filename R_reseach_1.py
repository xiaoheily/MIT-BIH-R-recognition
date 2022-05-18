import wfdb
import matplotlib.pyplot as plt
import pywt
import numpy as np


# 读取数据
def read_ecg_data(filePath, channel_names):
    record = wfdb.rdrecord(filePath, channel_names=[channel_names])  # 读取ECG信号
    print('导联线条数:')
    print(record.n_sig)  # 查看导联线条数
    print('信号名称（列表）')
    print(record.sig_name)  # 查看信号名称（列表）
    return record


# 绘制心电图
def draw_ecg_R(record):
    plt.plot(record)  # 绘制心电信号
    plt.show()


def get_modelmax(data):     # 获取小尺度校正后的模极大值
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet="db3", level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs    # 九层
    cD1_max, cD1_cordint = model_max(cD1[0:600], 1)   # 计算第一层极大值及其位置
    cD2_max, cD2_cordint = model_max(cD2[0:600], 2)  # 计算第二层极大值及其位置
    cD3_max, cD3_cordint = model_max(cD3[0:600], 3)  # 计算第三层极大值及其位置
    cD4_max, cD4_cordint = model_max(cD4[0:600], 4)  # 计算第四层极大值及其位置
    cD5_max, cD5_cordint = model_max(cD5[0:600], 5)  # 计算第五层极大值及其位置

    cD2_cordintr = screen(cD1, cD2, cD1_cordint, cD2_cordint)
    cD3_cordintr = screen(cD2, cD3, cD2_cordintr, cD3_cordint)
    cD4_cordintr = screen(cD3, cD4, cD3_cordintr, cD4_cordint)
    cD5_cordintr = screen(cD4, cD5, cD4_cordintr, cD5_cordint)

    return cD5_cordintr


def screen(cD1, cD2, cD1_cordint, cD2_cordint):    # 筛选模极大值，清除非奇异点位置
    cD2_cordintr = []
    for i in cD1_cordint:  # 通过隔层传播阈值，筛选下一层模极大值
        for j in cD2_cordint:
            if abs(i - j) < 50 and cD2[j] * cD1[i] > 0:
                cD2_cordintr.append(j)
    cD2_cordintr = sorted(list(set(cD2_cordintr)))
    return cD2_cordintr


def model_max(date, level):     #取模极大值
    cA9_max = date
    threshold = 0  # 阈值
    coordinates = []   # 极值坐标
    level = 2**(4+level)

    y = sorted(abs(date))  # 取阈值
    for i in range(len(y) - level, len(y)):
        threshold = threshold + y[i]
    threshold = threshold / level

    for i in range(1, len(cA9_max) - 2):   # 计算极大值
        if cA9_max[i] < date[i+1] and cA9_max[i+1] > date[i+2] or cA9_max[i] > date[i+1] and cA9_max[i+1] < date[i+2]:
            continue
        else:
            cA9_max[i+1] = 0

    for i in range(1, len(cA9_max) - 2):   # 取极大值
        if abs(cA9_max[i]) < threshold:
            cA9_max[i] = 0
        else:
            coordinates.append(i)
    return abs(cA9_max), coordinates


def checkout(date, modelmax, ecgnum):      # 检验筛选出的奇异点是否为R波波峰
    threshold = 0  # 阈值
    R_xloct = []  # R波位置
    R_yloct = []
    ecgval = []   # 奇异点对应心电图的值

    for i in range(1, len(modelmax)-1):   # 获取奇异点对应心电图的值
        ecgval.append(date[i])
    y = sorted(ecgval)  # 取阈值

    for i in range((len(y) - 4*ecgnum), len(y)):
        threshold = threshold + y[i]
    threshold = threshold / (ecgnum*4)

    for i in modelmax:                     # 校正
        if date[i] > threshold and date[i] > 0:
            R_xloct.append(i)
            R_yloct.append(date[i])

    for i in R_xloct:
        try:
            if abs(date[i] - date[i + 1]) < 10:
                if date[i] > date[i+1]:
                    R_xloct.remove(i+1)
                    R_yloct.remove(date[i+1])
                else:
                    R_xloct.remove(i)
                    R_yloct.remove(date[i])
        except:
            print()
    return R_xloct, R_yloct


def main():
    filePath = '../MIT-BIH R_recognition/ecg_data/101'
    Channel_Name = 'MLII'
    record = read_ecg_data(filePath, Channel_Name)

    record_cut = record.p_signal[0:600]    # 截取其中一小段实现, 长度为600

    data = record.p_signal.flatten()
    data_cut = data[0:600]

    modelmax = get_modelmax(data=data)
    x, y = checkout(date=data_cut, modelmax=modelmax, ecgnum=2)
    plt.scatter(x, y, s=50, c='r')  # s为点的大小

    draw_ecg_R(record_cut)


if __name__ == '__main__':
    main()
