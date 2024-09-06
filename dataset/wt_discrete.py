import pickle

import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
import pywt
import datetime

base_dir = ''
data_path = ''

fs = 100
sample = fs * 60  # 1 min's sample points

before = 2  # forward interval (min)
after = 2  # backward interval (min)
hr_min = 20
hr_max = 300

num_worker = 10  # Setting according to the number of CPU cores

level = 1 # wavelet decomposition level


def worker(name, labels, wavelet_name):
    signals = wfdb.rdrecord(os.path.join(data_path, name), channels=[0]).p_signal[:, 0]
    print_current_time()
    print(name, ': ', end='')
    print('label length:', len(labels))
    X1, X2, X3, X4, X5, X6 = [], [], [], [], [], []  # 5 levels of wavelet decomposition: X1 is low frequency, X2~X6 is high frequency
    y = []
    for j in range(len(labels)):
        if j % 20 == 0:
            print(j, end=' ')
        if j < before or \
                (j + 1 + after) > len(signals) / float(sample):
            continue
        signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
        signal_processed, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.03 * fs),
                                                  frequency=[0.05, 45], sampling_rate=fs)
        
        coeffs = pywt.wavedec(signal_processed, wavelet_name, level=level)

        X1.append(coeffs[0])
        X2.append(coeffs[1])
        X3.append(coeffs[2])
        X4.append(coeffs[3])
        X5.append(coeffs[4])
        X6.append(coeffs[5])
        y.append(0. if labels[j] == 'N' else 1.)

    return X1, X2, X3, X4, X5, X6, y, name


def print_current_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n[Current Time]", formatted_time)


if __name__ == "__main__":
    print('start')
    wavelet_name = 'sym4'

    train_list = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
                  "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
                  "b01", "b02", "b03", "b04", "b05",
                  "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"]
     
    for i in range(len(train_list)):
        labels = wfdb.rdann(os.path.join(data_path, train_list[i]), extension="apn").symbol
        X1, X2, X3, X4, X5, X6, y, sub = worker(train_list[i], labels, wavelet_name)
        apnea_ecg = dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5, X6=X6, y=y)
        with open(os.path.join(base_dir, sub + ".pkl"), "wb") as f:
            pickle.dump(apnea_ecg, f, protocol=2)
    print("\nok!")

    test_list = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]

    for i in range(len(test_list)):
        labels = wfdb.rdann(os.path.join(data_path, test_list[i]), extension="apn").symbol
        X1, X2, X3, X4, X5, X6, y, sub = worker(test_list[i], labels, wavelet_name)
        apnea_ecg = dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5, X6=X6, y=y)
        with open(os.path.join(base_dir, sub + ".pkl"), "wb") as f:
            pickle.dump(apnea_ecg, f, protocol=2)
    print("\nok!")