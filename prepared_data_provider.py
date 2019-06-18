import numpy as np

_config = {
    "channel_number": None,
    "freq_samples": None,
    "test_size": None,
    "train_size": None,
    "number_of_files": None
}
def config(channel_number, freq_samples, test_size, train_size, number_of_files):
    _config["channel_number"] = channel_number
    _config["freq_samples"] = freq_samples
    _config["test_size"] = test_size
    _config["train_size"] = train_size
    _config["number_of_files"] = number_of_files
def calculate_emotion(valence, arousal):
    if arousal >= 3 and valence >= 3 :
        return np.array([1,0,0])
    elif arousal >= 3 and valence < 3 :
        return np.array([0,1,0])
    else :
        return np.array([0,0,1])

def get_prepared_data(processed_data, data, test_start, test_x, test_y):
    train_elements = int(_config["train_size"]/_config["number_of_files"])
    input_train = np.zeros(shape=(train_elements, _config["channel_number"], _config["freq_samples"]))
    output_train = []
    i = test_start
    processed_data = np.array(processed_data).reshape(-1,_config["freq_samples"])
    for j in range(len(data)):
        (block_eeg, single_eval) = data[j]
            
        if j < train_elements:
            output_train.append(calculate_emotion(single_eval[0], single_eval[1]))
            input_train[j][0:_config["channel_number"]][0:_config["freq_samples"]] = processed_data[0:_config["channel_number"]][0:_config["freq_samples"]]
        else:
            test_y.append(calculate_emotion(single_eval[0], single_eval[1]))
            test_x[i][0:_config["channel_number"]][0:_config["freq_samples"]] = processed_data[0:_config["channel_number"]][0:_config["freq_samples"]]
            i+=1

    output_train = np.array(output_train)
    return (input_train, output_train)