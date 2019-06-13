from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum 

class Emotion(Enum):
    positive_exciting = 1
    calm = 2
    negative_exciting = 3

def calculate_emotion(valence, arousal) -> Emotion :
    if arousal >= 3 and valence >= 3 :
        return np.array([1,0,0])
    elif arousal >= 3 and valence < 3 :
        return np.array([0,1,0])
    else :
        return np.array([0,0,1])

def process_batch(data):
    for i in range(len(data)):
        print(f"Processing data segment {i}/{len(data)}")
        for j in range(len(data[i][0])):
            print(f"Processing data from electrode {j}/{len(data[i][0])}")
            data[i][2].append(time_to_freq_domain(data[i][0][j]))
    return data


def time_to_freq_domain(data):
    magnitudes = np.abs(fftpack.rfft(data))
    frequencies = fftpack.fftfreq(len(magnitudes)) * 1024
    start_freq = 0.2
    end_freq = 30.0

    start_freq_index = int(len(frequencies) / 2 / max(frequencies) * start_freq) + 1
    end_freq_index = int(len(frequencies) / 2 / max(frequencies) * end_freq) + 1
    magnitudes = magnitudes[start_freq_index:end_freq_index]
    frequencies = frequencies[start_freq_index:end_freq_index]

    compressed_magnitudes = []
    temp_magnitudes_to_compress = []
    freq_step = 0.1
    current_freq = start_freq + freq_step

    for i in range(len(frequencies)):
        if frequencies[i] < current_freq:
            temp_magnitudes_to_compress.append(magnitudes[i])
        else:
            compressed_magnitudes.append(sum(temp_magnitudes_to_compress) / len(temp_magnitudes_to_compress))
            temp_magnitudes_to_compress = []
            current_freq += freq_step

    # Normalize frequencies against max, comment if not needed.
    compressed_magnitudes = [float(i) / max(compressed_magnitudes) for i in compressed_magnitudes]

    # Display bar chart of frequencies against their corresponding magnitudes.
    # compressed_frequencies = np.arange(start=start_freq, stop=end_freq, step=freq_step).tolist()
    # compressed_frequencies = compressed_frequencies[:len(compressed_magnitudes) - len(compressed_frequencies)]
    # plt.bar(compressed_frequencies, compressed_magnitudes, freq_step)
    # plt.xlabel('Frequency in Hertz [Hz]')
    # plt.ylabel('Frequency Domain (Spectrum) Magnitude')
    # plt.show()
    return compressed_magnitudes
