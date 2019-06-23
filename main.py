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

#model_conv = Sequential()
#model_conv.add(Conv1D(64, 3, activation='relu', input_shape=(channel_number, freq_samples)))
##model_conv.add(Conv1D(64, 3, activation='relu'))
#model_conv.add(MaxPooling1D(3))
#model_conv.add(Conv1D(128, 3, activation='relu'))
#model_conv.add(Conv1D(128, 3, activation='relu'))
#model_conv.add(GlobalAveragePooling1D())
#model_conv.add(Dropout(0.5))
#model_conv.add(Dense(max_output_size, activation='sigmoid'))
#model_conv.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

model_conv = Sequential()
model_conv.add(Conv1D(128, 3, activation='relu', input_shape=(channel_number, freq_samples)))
model_conv.add(keras.layers.Flatten())
model_conv.add(Dense(max_output_size, activation='sigmoid'))
model_conv.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# load data
train_x = []
train_y = []
for i in range(number_of_files):
    data = cvapr_data.load_data_from_files(i)[:data_per_file]
    (temp_train_x, temp_train_y) = prepared_data_provider.get_prepared_data(data, int(i * (test_size / number_of_files)), test_x, test_y)
    train_x.append(temp_train_x)
    train_y.append(temp_train_y)

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
