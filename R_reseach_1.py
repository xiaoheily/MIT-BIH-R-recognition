import wfdb
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.signal import medfilt


# 读取数据
def read_ecg_data(filePath, channel_names, sampfrom, sampto):
    record = wfdb.rdrecord(filePath, channel_names=[channel_names], sampfrom=sampfrom, sampto=sampto, physical=True)  # 读取ECG信号
    print('导联线条数:')
    print(record.n_sig)  # 查看导联线条数
    print('信号名称（列表）')
    print(record.sig_name)  # 查看信号名称（列表）
    annotation = wfdb.rdann(filePath, 'atr')    # 查atr文件中R波位置
    return record, annotation


# 绘制心电图
def draw_ecg_R(record):
    plt.plot(record)  # 绘制心电信号
    plt.title("R wave in ECG data")
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.show()


def baseline(data):       # 基线漂移校正
    fliter = int(0.8 * 360)
    Give_up_size = int(fliter / 2)

    ECG_baseline = medfilt(data, fliter + 1)
    data = data - ECG_baseline
    return data


def denoise(data):    # 小波去噪预处理
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=4)
    cA4, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)

    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def get_modelmax(data, sampling_rate):     # 获取小波变换连续尺度上的模极大值
    wavename = 'mexh'       # 采用的小波基
    scales = [2, 3, 4, 5, 6]  # 尺度

    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    scale1_axis = model_max(cwtmatr[0])
    scale2_axis = model_max(cwtmatr[1])
    scale3_axis = model_max(cwtmatr[2])
    scale4_axis = model_max(cwtmatr[3])
    scale5_axis = model_max(cwtmatr[4])

    scale4_axisr = screen(cwtmatr[4], cwtmatr[3], scale5_axis, scale4_axis)
    scale3_axisr = screen(cwtmatr[3], cwtmatr[2], scale4_axisr, scale3_axis)
    scale2_axisr = screen(cwtmatr[2], cwtmatr[1], scale3_axisr, scale2_axis)
    scale1_axisr = screen(cwtmatr[1], cwtmatr[0], scale2_axisr, scale1_axis)

    return scale1_axisr


def screen(cD1, cD2, cD1_cordint, cD2_cordint):    # 筛选模极大值，清除非奇异点位置
    cD2_cordintr = []
    for i in cD1_cordint:  # 通过隔层传播阈值，筛选下一层模极大值
        for j in cD2_cordint:
            if abs(i - j) < 5 and cD2[j] * cD1[i] > 0:
                cD2_cordintr.append(j)
    cD2_cordintr = sorted(list(set(cD2_cordintr)))
    return cD2_cordintr


def model_max(date):     # 取模极大值
    cA9_max = date
    threshold = 0  # 阈值
    coordinates = []   # 极值坐标
    level = int(len(date)*0.05)
    """
    level = 2 ** (4 + level)
    if level > len(date):
        level = len(date)
    """
    y = sorted(abs(date))  # 取阈值
    for i in range(len(y) - level, len(y)):
        threshold = threshold + y[i]
    threshold = threshold / level

    for i in range(0, len(cA9_max) - 2):   # 计算极大值
        if cA9_max[i] < date[i+1] and cA9_max[i+1] > date[i+2] or cA9_max[i] > date[i+1] and cA9_max[i+1] < date[i+2]:
            continue
        else:
            cA9_max[i+1] = 0

    for i in range(0, len(cA9_max)):   # 取极大值
        if abs(cA9_max[i]) < threshold:
            cA9_max[i] = 0
        else:
            coordinates.append(i)

    return coordinates


def checkout(data, modelmax, ecgnum):      # 检验筛选出的奇异点是否为R波波峰,
    threshold = 0  # 阈值
    R_xloct = []  # R波位置
    R_yloct = []
    ecgval = []   # 奇异点对应心电图的模值

    for i in modelmax:                       # 获取奇异点对应心电图的值
        ecgval.append(abs(data[i]))
    # plt.scatter(modelmax, ecgval, c='r')

    y = sorted(ecgval)  # 取阈值
    for i in range(0, len(y)):
        threshold = threshold + y[i]
    threshold = threshold / (ecgnum*3)

    for i in modelmax:                     # 利用阈值进行校正
        if abs(data[i]) > threshold:
            R_xloct.append(i)
            R_yloct.append(data[i])

    for i in range(0, len(R_xloct)-1):         # 去除掉同处于一个R波的奇异点
        if abs(R_xloct[i] - R_xloct[i+1]) < 20:
            if R_yloct[i] < R_yloct[i+1]:
                R_xloct[i] = 0
                R_yloct[i] = 0
            else:
                R_xloct[i+1] = 0
                R_yloct[i+1] = 0

    while 0 in R_xloct:
        R_xloct.remove(0)
        R_yloct.remove(0)

    R_xloctr = []                # 存储校正后的R波波峰
    for i in R_xloct:             # 校正R波波峰位置
        index_cpar = [data[i-2], data[i-1], data[i], data[i+1], data[i+2]]
        max_index = index_cpar.index(max(index_cpar, key=abs))
        R_xloctr.append(i-2 + max_index)

    return R_xloctr


def selData(annotation, label):            # 读取atr文件中的R波位置
    a = annotation.symbol   # 数据中的五类标签
    f = [k for k in range(len(a)) if a[k] == label]  # 找到对应标签R波位置索引
    R_pos = annotation.sample[f]     # R波坐标

    return R_pos


def error_analysis(R_pos, x):         # 误差分析
    R_pos = list(R_pos)
    print("正常心拍总数", len(R_pos))
    print("检测出的心拍总数", len(x))
    xr = []
    for i in x:
        xr.append(i - 1)
        xr.append(i + 1)
        xr.append(i)
    xr = set(xr)

    same = [i for i in R_pos if i in xr]  # 测出的正确的R波位置
    lack_identify = [i for i in R_pos if i not in xr]  # 漏检

    print("正确心拍数", len(same))
    print("误检心拍数", len(x)-len(same))
    print("漏检心拍数", len(lack_identify))
    print("误检率", (len(x)-len(same) + len(lack_identify))/len(R_pos))


def main():
    start = 0  # 截取一段实现
    stop = 650000
    sampling_rate = 360    # 采样频率
    filePath = '../MIT-BIH R_recognition/ecg_data/200'
    Channel_Name = 'MLII'    # 获取MLⅡ导联

    record, annotation = read_ecg_data(filePath, Channel_Name, sampfrom=start, sampto=stop)    # 读取ECG信号
    data = record.p_signal.flatten()
    R_pos = selData(annotation, 'N')    # 读取R波位置
    ecgnum = len(R_pos)  # 截取数据中ecg信号波数
    # ecgnum = 8  # 截取数据中ecg信号波数

    datar = baseline(data)   # 基线漂移校正
    datarr = denoise(datar)      # 工频干扰去除
    data_cut = data[start:stop]      # 截取一段绘制，展示效果

    modelmax_axis = get_modelmax(datarr, sampling_rate)

    x = checkout(data=data_cut, modelmax=modelmax_axis, ecgnum=ecgnum)  # ecgnum为采样时间内波数
    # plt.scatter(x, data[x], s=50, c='r')  # s为点的大小
    # draw_ecg_R(data_cut)

    error_analysis(R_pos, x)   # 误差分析


if __name__ == '__main__':
    main()
