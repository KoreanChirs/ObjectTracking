from data_loader import organize_data, is_organize_data
from physics_model import physics_based_prediction
import numpy as np
from collections import defaultdict
from iou_cal import iou_2d_cal

file_path_1 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-01/gt/gt.txt"
file_path_2 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-02/gt/gt.txt"
file_path_3 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-03/gt/gt.txt"
file_path_4 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-05/gt/gt.txt"

paths = [file_path_1,file_path_2,file_path_3,file_path_4]

FPS = 25
dt = 1/FPS

def inference_one_fct(file_path):
    predictions = defaultdict(list)
    is_organize_data(file_path)
    tracking_dict = organize_data(file_path)
    for i in tracking_dict:
        tracking_dict[i] = np.array(tracking_dict[i])
    for track_id in tracking_dict:
        top_left_x = np.array(physics_based_prediction(tracking_dict[track_id][:,0],dt))
        top_left_y = np.array(physics_based_prediction(tracking_dict[track_id][:,1],dt))
        width = np.array(physics_based_prediction(tracking_dict[track_id][:,2],dt))
        height = np.array(physics_based_prediction(tracking_dict[track_id][:,3],dt))
        predictions[track_id] = np.array([top_left_x,top_left_y,width,height]).T
    
    IOU_results = [iou_2d_cal(predictions[idx],tracking_dict[idx]) for idx in tracking_dict]
    IOU_results = list(filter(lambda x : x is not None, IOU_results))
    return sum(IOU_results)/len(IOU_results)

def inference(file_paths):
    for path in file_paths:
        print("Iou avg: ",inference_one_fct(path))

inference(paths)
