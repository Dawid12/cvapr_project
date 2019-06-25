import numpy as np
from data_processor import scale, standardize

_config = {
    "channel_number": None,
    "freq_samples": None,
    "test_size": None,
    "train_size": None,
    "number_of_files": None
}


def config(channel_number, freq_samples, test_size, train_size, number_of_files, test_to_train):
    _config["channel_number"] = channel_number
    _config["freq_samples"] = freq_samples
    _config["test_size"] = test_size
    _config["train_size"] = train_size
    _config["number_of_files"] = number_of_files
    _config["test_to_train"] = test_to_train


def calculate_emotion(valence, arousal):
    if arousal >= 3 and valence >= 3 :
        return np.array([1,0,0])
    elif arousal >= 3 and valence < 3 :
        return np.array([0,1,0])
    else :
        return np.array([0,0,1])


def get_prepared_data(data, test_start, test_x, test_y):
    train_elements = int(_config["train_size"]/_config["number_of_files"])
    input_train = np.zeros(shape=(train_elements, _config["channel_number"], _config["freq_samples"]))
    output_train = []
    i = test_start
    for j in range(len(data)):
        processed_data = data[j].power_spectrum(1, 100, 1)
        for channel_data in processed_data:
            #scale(channel_data)
            standardize(channel_data)
        block_eeg = []
        for channel_data in processed_data[0]:
            for sample in channel_data:
                block_eeg.append(sample)
        single_eval = data[j].evaluation

        if j < train_elements:
            output_train.append(calculate_emotion(single_eval[0], single_eval[1]))
            input_train[j][0:_config["channel_number"]][0:_config["freq_samples"]] = processed_data[0][0:_config["channel_number"]][0:_config["freq_samples"]]
        else:
            test_y.append(calculate_emotion(single_eval[0], single_eval[1]))
            test_x[i][0:_config["channel_number"]][0:_config["freq_samples"]] = processed_data[0][0:_config["channel_number"]][0:_config["freq_samples"]]
            i+=1

    output_train = np.array(output_train)
    return (input_train, output_train)
