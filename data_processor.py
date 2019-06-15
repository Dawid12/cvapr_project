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
    """ Processes batch of raw data.
        A batch is an array of five arrays corresponding to five blocks in a session.
        Each of these arrays consists of 71 arrays representing EEG data for each electrode."""
    print("Processing data...")
    processed_data = []
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            a=5
            processed_data.append(time_to_freq_domain(data[i][0][j]))
    a = 5
    print("Done")
    return processed_data


def time_to_freq_domain(data):
    magnitudes = np.abs(fftpack.rfft(data))
    frequencies = fftpack.fftfreq(len(magnitudes)) * 1024
    start_freq = 1.0
    end_freq = 100.0
    freq_step = 0.1
    expected_results_list_length = (end_freq - start_freq) / freq_step - 1

    start_freq_index = int(len(frequencies) / 2 / max(frequencies) * start_freq) + 1
    end_freq_index = int(len(frequencies) / 2 / max(frequencies) * end_freq) + 1
    magnitudes = magnitudes[start_freq_index:end_freq_index]
    frequencies = frequencies[start_freq_index:end_freq_index]

    compressed_magnitudes = []
    temp_magnitudes_to_compress = []
    current_freq = start_freq + freq_step

    for i in range(len(frequencies)):
        if frequencies[i] < current_freq:
            temp_magnitudes_to_compress.append(magnitudes[i])
        else:
            compressed_magnitudes.append(sum(temp_magnitudes_to_compress) / len(temp_magnitudes_to_compress))
            temp_magnitudes_to_compress = []
            current_freq += freq_step

    # Trim and normalize frequencies against max, comment if not needed.
    compressed_magnitudes = compressed_magnitudes[0:int(expected_results_list_length) - 1]
    compressed_magnitudes = [float(i) / max(compressed_magnitudes) for i in compressed_magnitudes]

    # Display bar chart of frequencies against their corresponding magnitudes.
    # compressed_frequencies = np.arange(start=start_freq, stop=end_freq, step=freq_step).tolist()
    # compressed_frequencies = compressed_frequencies[:len(compressed_magnitudes) - len(compressed_frequencies)]
    # plt.bar(compressed_frequencies, compressed_magnitudes, freq_step)
    # plt.xlabel('Frequency in Hertz [Hz]')
    # plt.ylabel('Frequency Domain (Spectrum) Magnitude')
    # plt.show()
    return compressed_magnitudes
