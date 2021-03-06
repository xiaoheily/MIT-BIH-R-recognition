import wfdb
import matplotlib.pyplot as plt
import numpy as np


# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata



# 画心电图及其R波位置
def draw_ecg_R(record, annotation):
    plt.plot(record.p_signal)  # 绘制心电信号
    R_v = record.p_signal[annotation.sample]  # 获取R波峰值
    print(len(R_v))
    plt.plot(annotation.sample[1::], R_v[1::], 'or')  # 绘制R波
    plt.title('Raw_ECG And R Position')
    plt.show()



def selData(annotation, label):
    a = annotation.symbol   # 数据中的五类标签
    f = [k for k in range(len(a)) if a[k] == label]  # 找到对应标签R波位置索引
    R_pos = annotation.sample[f]     # R波坐标

    return R_pos


# 读取心电图数据
def read_ecg_data(filePath, channel_names):
    '''
    读取心电信号文件
    sampfrom: 设置读取心电信号的 起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的 结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    channel_names：设置设置读取心电信号名字，必须是列表，channel_names=['MLII']表示读取MLII导联线
    channels：设置读取第几个心电信号，必须是列表，channels=[0, 3]表示读取第0和第3个信号，注意信号数不确定
    record = wfdb.rdrecord('../ecg_data/102', sampfrom=0, sampto = 1500) # 读取所有通道信号
    record = wfdb.rdrecord('../ecg_data/203', sampfrom=0, sampto = 1500,channel_names=['MLII']) # 仅仅读取“MLII”信号
    record = wfdb.rdrecord('../ecg_data/101', sampfrom=0, sampto=3500, channels=[0]) # 仅仅读取第0个信号（MLII）
    print(type(record)) # 查看record类型
    print(dir(record)) # 查看类中的方法和属性
    print(record.p_signal) # 获得心电导联线信号，本文获得是MLII和V1信号数据
    print(record.n_sig) # 查看导联线条数
    print(record.sig_name) # 查看信号名称（列表），本文导联线名称['MLII', 'V1']
    print(record.fs) # 查看采用率
    '''

    record = wfdb.rdrecord(filePath, channel_names=[channel_names])
    print('导联线条数:')
    print(record.n_sig)  # 查看导联线条数
    print('信号名称（列表）')
    print(record.sig_name)  # 查看信号名称（列表），本文导联线名称['MLII', 'V1']

    '''
    读取注解文件
    sampfrom: 设置读取心电信号的 起始位置，sampfrom=0表示从0开始读取，默认从0开始
    sampto：设置读取心电信号的 结束位置，sampto = 1500表示从1500出结束，默认读到文件末尾
    print(type(annotation)) # 查看annotation类型
    print(dir(annotation))# 查看类中的方法和属性
    print(annotation.sample) # 标注每一个心拍的R波的尖锋位置，与心电信号对应
    annotation.symbol  #标注每一个心拍的类型N，L，R等等
    print(annotation.ann_len) # 被标注的数量
    print(annotation.record_name) # 被标注的文件名
    print(wfdb.show_ann_labels()) # 查看心拍的类型
    '''
    annotation = wfdb.rdann(filePath, 'atr')
    # print(annotation.symbol)
    return record, annotation


def main():
    filePath = '../MIT-BIH R_recognition/ecg_data/217'
    Channel_Name = 'MLII'
    record, annotation = read_ecg_data(filePath, Channel_Name)
    #     draw_ecg(record.p_signal)

    draw_ecg_R(record, annotation)
    R_pos = selData(annotation, 'N')
    # plt.plot(res[20])  # 随便绘制一个所截取的心电图
    np.save('test.out', R_pos)


    # plt.show()


if __name__ == "__main__":
    main()
