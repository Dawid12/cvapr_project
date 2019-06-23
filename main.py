import numpy as np
import random
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D,Conv2D, GlobalAveragePooling1D, MaxPooling1D
import cvapr_data
import data_processor
import prepared_data_provider
#configuration
random_seed = 42
channel_number = 64
freq_samples = 100
max_output_size = 3
data_per_file = 30
test_to_train = 0.2
number_of_files = 11
test_size = int(test_to_train*(number_of_files*data_per_file))
train_size = int((1-test_to_train)*(number_of_files*data_per_file))
test_x = np.zeros(shape=(test_size, channel_number, freq_samples))
test_y = []

enterface06_EMOBRAIN_path = "C:\\Users\\pawel\\Documents\\Studia\\CVaPR\\Projekt"
cvapr_data.configure(enterface06_EMOBRAIN_path, 254)
prepared_data_provider.config(channel_number, freq_samples, test_size, train_size, number_of_files, test_to_train)
random.seed(random_seed)

# load data
train_x = []
train_y = []
for i in range(number_of_files):
    data = cvapr_data.load_data_from_files(i)[:data_per_file]
    #data = cvapr_data.load_data_from_files(i, low_freq = 1, high_freq = 100)[:data_per_file]
    (temp_train_x, temp_train_y) = prepared_data_provider.get_prepared_data(data, int(i * (test_size / number_of_files)), test_x, test_y)
    train_x.append(temp_train_x)
    train_y.append(temp_train_y)

scores = []

def create_model_1():  # 46.54%, 44,59% from 7 for standardize, power post split, no filtering, random 42
    model_conv = Sequential()
    model_conv.add(Conv1D(64, 11, activation='relu', input_shape=(channel_number, freq_samples)))
    model_conv.add(Conv1D(128, 7, activation='relu'))
    model_conv.add(Conv1D(256, 3, activation='relu'))
    model_conv.add(keras.layers.Flatten())
    model_conv.add(Dense(max_output_size, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model_conv


def create_model_2():  # 43,30%, 43,07% from 7 for standardize, power post split, no filtering, random 42
    model_conv = Sequential()
    model_conv.add(Conv1D(128, 3, activation='relu', input_shape=(channel_number, freq_samples)))
    model_conv.add(keras.layers.Flatten())
    model_conv.add(Dense(max_output_size, activation='sigmoid'))
    model_conv.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model_conv

def create_model_3():  # 48,05%, 47,40%, 46,10% from 7 for standardize, power post split, no filtering, random 42
    model_conv = Sequential()
    model_conv.add(Conv1D(128, 3, activation='relu', input_shape=(channel_number, freq_samples)))
    model_conv.add(keras.layers.Flatten())
    model_conv.add(Dense(max_output_size, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model_conv

def create_model_4():  # 48,70%, 48,27%, 47,84% from 7 for standardize, power post split, no filtering, random 42
    model_conv = Sequential()
    model_conv.add(Conv1D(256, 7, activation='relu', input_shape=(channel_number, freq_samples)))
    model_conv.add(keras.layers.Flatten())
    model_conv.add(Dense(max_output_size, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model_conv

def create_model_5():  # 48,70%, 48,27%, 47,84%, 42,00% from 7 for standardize, power post split, no filtering, random 42
    model_conv = Sequential()
    model_conv.add(Conv1D(256, 7, activation='relu', input_shape=(channel_number, freq_samples)))
    model_conv.add(Conv1D(256, 7, activation='relu'))
    model_conv.add(keras.layers.Flatten())
    model_conv.add(Dense(max_output_size, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model_conv

for _ in range(7):

    model_conv = create_model_5()

    # train
    target_accuracy = 0.6
    no_improvements_iterations_limit = 10
    continue_training = True
    no_improvements_iterator = 0
    last_improved_accuracy = 0
    while continue_training:
        train_xy = list(zip(train_x, train_y))
        random.shuffle(train_xy)
        train_x[:], train_y[:] = zip(*train_xy)
        for i in range(number_of_files):
            model_conv.fit(train_x[i], train_y[i], batch_size=len(train_x[i]), epochs=5)

        # test & check train end criteria
        test_y = np.array(test_y)
        score = model_conv.evaluate(test_x, test_y, batch_size=len(test_y))
        if score[1] > target_accuracy or no_improvements_iterator > no_improvements_iterations_limit:
            continue_training = False
        elif score[1] > last_improved_accuracy:
            last_improved_accuracy = score[1]
            no_improvements_iterator = 0
        else:
            no_improvements_iterator += 1

    #test
    test_y = np.array(test_y)
    score = model_conv.evaluate(test_x, test_y, batch_size=len(test_y))
    print(score)

    scores.append(score)

scores = np.array(scores)
model_conv.summary()
print(scores)
print(scores[:, 1].mean())
print(scores[:, 1].std())
