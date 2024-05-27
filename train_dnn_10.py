import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from data_loader import organize_data
from keras.callbacks import EarlyStopping

file_path_1 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-01/gt/gt.txt"
file_path_2 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-02/gt/gt.txt"
file_path_3 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-03/gt/gt.txt"
file_path_4 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-05/gt/gt.txt"

# Example training data dictionary
training_data_1 = organize_data(file_path_1)
training_data_2 = organize_data(file_path_2)
training_data_3 = organize_data(file_path_3)
training_data_4 = organize_data(file_path_4)

training_datas = [training_data_1,training_data_2,training_data_3,training_data_4]

# Define and compile the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(12,)),  # Input layer, assuming flattened sequence of 3 timesteps with 4 features each
    Dense(12, activation='relu'),
    Dense(4)  # Output layer with 4 units
])
model.compile(optimizer='adam', loss='mse')

train_x = []
train_y = []

# Flatten the sequence data into one input array per sample
for training_data in training_datas:
    for object_id, sequence in training_data.items():
        if(len(sequence) < 13):
            continue
        for i in range(12,len(sequence)):
            X_train = np.concatenate((sequence[i-12], sequence[i-11], sequence[i-10]))
            y_train = sequence[i]
            train_x.append(X_train)
            train_y.append(y_train)
train_x = np.array(train_x) / 100
train_y = np.array(train_y) / 100

# Process validation data from training_data_2 in the same way
val_x = []
val_y = []
for object_id, sequence in training_data_2.items():
    if(len(sequence) < 13):
        continue
    for i in range(12,len(sequence)):
        X_val = np.concatenate((sequence[i-12], sequence[i-11], sequence[i-10]))
        y_val = sequence[i]
        val_x.append(X_val)
        val_y.append(y_val)

val_x = np.array(val_x) / 100
val_y = np.array(val_y) / 100

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model.fit(train_x, train_y, epochs=100, batch_size=8, verbose=1, validation_data=(val_x, val_y), callbacks=[early_stopping])
model.save('custom_DNN_10.h5')
