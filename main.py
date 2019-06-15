import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D,Conv2D, GlobalAveragePooling1D, MaxPooling1D
import cvapr_data
import data_processor
#configuration
enterface06_EMOBRAIN_path = "C:\\Users\\pawel\\Documents\\Studia\\CVaPR\\Projekt"
cvapr_data.configure(enterface06_EMOBRAIN_path, 254)
channel_number = 71
freq_samples = 988
max_output_size = 3
data_per_file = 5
model_conv = Sequential()
model_conv.add(Conv1D(64, 3, activation='relu', input_shape=(channel_number, freq_samples)))
model_conv.add(Conv1D(64, 3, activation='relu'))
model_conv.add(MaxPooling1D(3))
model_conv.add(Conv1D(128, 3, activation='relu'))
model_conv.add(Conv1D(128, 3, activation='relu'))
model_conv.add(GlobalAveragePooling1D())
model_conv.add(Dropout(0.5))
model_conv.add(Dense(max_output_size, activation='sigmoid'))
model_conv.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
input_test = []
output_test = []
#data processing
for i in range(2, 11):
    output_train = []
    data = cvapr_data.load_data_from_files(i)[:data_per_file]
    processed_data = data_processor.process_batch(data)
    train = np.zeros(shape=(len(data), channel_number, freq_samples))
    for j in range(len(data)):
        (block_eeg, single_eval) = data[j]
        output_train.append(data_processor.calculate_emotion(single_eval[0], single_eval[1]))
        train[j][0:channel_number][0:freq_samples] = processed_data[0:channel_number][0:freq_samples]
    input_test.append(train[int(0.8*data_per_file):int(data_per_file)])
    output_test.append(output_train[int(0.8*data_per_file):int(data_per_file)])
    output_train = np.array(output_train)
    model_conv.fit(train, output_train, batch_size=len(train), epochs=10)
#test -> jeszcze nie działa
input_test = np.array(input_test)
input_test = np.reshape(input_test, (len(input_test), channel_number, freq_samples))
output_test = np.array(output_test)
score = model_conv.evaluate(input_test, output_test, batch_size=len(input_test))
print(score)
