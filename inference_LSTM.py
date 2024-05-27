from data_loader import organize_data, is_organize_data
import numpy as np
from collections import defaultdict
from iou_cal import iou_2d_cal
from keras.models import load_model

file_path_1 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-01/gt/gt.txt"
file_path_2 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-02/gt/gt.txt"
file_path_3 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-03/gt/gt.txt"
file_path_4 = "D:/DGIST/4학년/Reinforcement Learning/Project/Main/MOT20Labels/train/MOT20-05/gt/gt.txt"

paths = [file_path_1, file_path_2, file_path_3, file_path_4]

model = load_model("custom_LSTM.h5")

def LSTM_based_prediction(input_seq, model, normalization_factor=100):
    prediction = []
    normalized_seq = input_seq / normalization_factor  # Normalize input sequence
    if len(normalized_seq) > 0:
        prediction.append([None, None, None, None])
    if len(normalized_seq) > 1:
        prediction.append(input_seq[0])
    if len(normalized_seq) > 2:
        prediction.append(input_seq[1])
    if len(normalized_seq) > 3:
        prediction.append(input_seq[2])
    for i in range(4, len(normalized_seq)):
        model_input = np.array([normalized_seq[i-3], normalized_seq[i-2], normalized_seq[i-1]]).reshape(1, 3, 4)
        model_output = model.predict(model_input, verbose=0)
        prediction.append(model_output[0] * normalization_factor)  # Denormalize predictions

    return np.array(prediction)

def inference_one_fct(file_path, model, normalization_factor=100):
    predictions = defaultdict(list)
    is_organize_data(file_path)
    tracking_dict = organize_data(file_path)
    for i in tracking_dict:
        tracking_dict[i] = np.array(tracking_dict[i])
    for track_id in tracking_dict:
        predictions[track_id] = LSTM_based_prediction(tracking_dict[track_id], model, normalization_factor)
        print(f"done. track_id: {track_id}")
        print(iou_2d_cal(predictions[track_id], tracking_dict[track_id]))

    IOU_results = [iou_2d_cal(predictions[idx], tracking_dict[idx]) for idx in tracking_dict]
    IOU_results = list(filter(lambda x: x is not None, IOU_results))
    return sum(IOU_results) / len(IOU_results)

def inference(file_paths, model):
    for path in file_paths:
        print("Iou avg: ", inference_one_fct(path, model))

inference(paths, model)

