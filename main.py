import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import cvapr_data
#data load
enterface06_EMOBRAIN_path = "C:\\Users\\Dawid\\Desktop\\cvapr\\data"
cvapr_data.configure(enterface06_EMOBRAIN_path, 254)
data = cvapr_data.load_data_from_files(*range(11))[:-20]
dataset_size = len(data)
max_input_size = 30000
max_output_size = 2
#for block_eeg,single_eval in data:
#    if len(block_eeg) > max_input_size:
#        max_input_size = len(block_eeg)
x_train = np.zeros(shape=(dataset_size,max_input_size))
y_train = np.zeros(shape=(dataset_size,max_output_size))
for i in range(len(data)):
    (block_eeg,single_eval) = data[i]
    #block_eeg_len = len(block_eeg)
    single_eval_len = len(single_eval)
    x_train[i][0:max_input_size] = block_eeg[0:max_input_size]
    y_train[i][0:single_eval_len] = single_eval
#split into train and test
x_test = x_train[int(0.8*dataset_size):int(dataset_size)]
y_test = y_train[int(0.8*dataset_size):int(dataset_size)]
x_train = x_train[0:int(0.8*dataset_size)-1]
y_train = y_train[0:int(0.8*dataset_size)-1]

#nn
model = Sequential()
model.add(Dense(500, input_dim=max_input_size, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=128)
score = model.evaluate(x_train, y_train, batch_size=128)
print(score)

#conv nn
x_train = np.expand_dims(x_train, axis=2)
model_conv = Sequential()
model_conv.add(Conv1D(64, 3, activation='relu', input_shape=(max_input_size, 1)))
model_conv.add(Conv1D(64, 3, activation='relu'))
model_conv.add(MaxPooling1D(3))
model_conv.add(Conv1D(128, 3, activation='relu'))
model_conv.add(Conv1D(128, 3, activation='relu'))
model_conv.add(GlobalAveragePooling1D())
model_conv.add(Dropout(0.5))
model_conv.add(Dense(max_output_size, activation='sigmoid'))

model_conv.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model_conv.fit(x_train, y_train, batch_size=16, epochs=10)
score = model_conv.evaluate(x_train, y_train, batch_size=16)
print(score)