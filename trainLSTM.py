import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from data_loader import organize_data
from keras.callbacks import EarlyStopping

file_path_1 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-01/gt/gt.txt"
file_path_2 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-02/gt/gt.txt"
file_path_3 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-03/gt/gt.txt"
file_path_4 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-05/gt/gt.txt"

# Example training data dictionary
training_data_1 = organize_data(file_path_1)
training_data_2 = organize_data(file_path_2)
training_data_3 = organize_data(file_path_3)
training_data_4 = organize_data(file_path_4)

training_datas = [training_data_1]

# Define and compile the model
model = Sequential([
    LSTM(units=8,input_shape=(3, 4),return_sequences=False),
    Dense(units=4)  # Output layer with 4 units
])
model.compile(optimizer='adam', loss='mse')

train_x = []
train_y = []

for training_data in training_datas:
    for object_id, sequence in training_data.items():
        if(len(sequence) < 5):
            continue
        for i in range(len(sequence) - 3):    
            X_train = []
            X_train.append(sequence[i])
            X_train.append(sequence[i+1])
            X_train.append(sequence[i+2])
            y_train = sequence[i+3]    
            X_train = np.array(X_train)
            y_train = np.array(y_train) 
            train_x.append(X_train)
            train_y.append(y_train)
train_x = np.array(train_x)/100
train_y = np.array(train_y)/100

# Processing validation data from training_data_4
val_x = []
val_y = []

for object_id, sequence in training_data_2.items():
    if(len(sequence) < 5):
        continue
    for i in range(len(sequence) - 3):    
        X_val = []
        X_val.append(sequence[i])
        X_val.append(sequence[i+1])
        X_val.append(sequence[i+2])
        y_val = sequence[i+3]    
        X_val = np.array(X_val)
        y_val = np.array(y_val)    
        val_x.append(X_val)
        val_y.append(y_val)

val_x = np.array(val_x) / 100
val_y = np.array(val_y) / 100

early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)

model.fit(train_x, train_y, epochs=3, batch_size=8, verbose=1,validation_data=(val_x, val_y), callbacks=[early_stopping])
model.save('custom_LSTM.h5')


