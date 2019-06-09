from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt


def process_batch(data):
    for i in range(len(data)):
        print(f"Processing data segment {i}/{len(data)}")
        for j in range(len(data[i][0])):
            print(f"Processing data from electrode {j}/{len(data[i][0])}")
            data[i][2].append(time_to_freq_domain(data[i][0][j]))
    return data


def time_to_freq_domain(data):
    x = np.abs(fftpack.rfft(data))
    freqs = fftpack.fftfreq(len(x)) * 1024
    startFreq = 0.2
    endFreq = 30

    startFreqIndex = int(len(freqs) / 2 / max(freqs) * startFreq)
    endFreqIndex = int(len(freqs) / 2 / max(freqs) * endFreq)
    x = x[startFreqIndex:endFreqIndex]
    freqs = freqs[startFreqIndex:endFreqIndex]

    # plt.bar(freqs, x, 1 / (endFreq - startFreq))
    # plt.xlabel('Frequency in Hertz [Hz]')
    # plt.ylabel('Frequency Domain (Spectrum) Magnitude')
    # plt.show()
    return x
