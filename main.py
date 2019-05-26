import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
import cvapr_data
enterface06_EMOBRAIN_path = "C:\\Users\\Dawid\\Desktop\\cvapr\\data"
cvapr_data.configure(enterface06_EMOBRAIN_path, 254)
data = cvapr_data.load_data_from_files(1,2,3,4,5,6,7,8,9,10)
dataset_size = len(data)
max_input_size = 0
max_output_size = 2
for block_eeg,single_eval in data:
    if len(block_eeg) > max_input_size:
        max_input_size = len(block_eeg)
x_train = np.zeros(shape=(dataset_size,max_input_size))
y_train = np.zeros(shape=(dataset_size,max_output_size))
for i in range(len(data)):
    (block_eeg,single_eval) = data[i]
    block_eeg_len = len(block_eeg)
    single_eval_len = len(single_eval)
    x_train[i][0:block_eeg_len] = block_eeg
    y_train[i][0:single_eval_len] = single_eval
#data_split_value = int(0.8*dataset_size)
#x_test = np.vsplit(x_train, data_split_value)
#y_test = np.vsplit(y_train, data_split_value)

model = Sequential()
model.add(Dense(1000, input_dim=max_input_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, batch_size=128)
score = model.evaluate(x_train, y_train, batch_size=128)
print(score)
#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)