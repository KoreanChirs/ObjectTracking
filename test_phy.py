from data_loader import test_data_loader
import numpy as np
from physics_model import physics_based_prediction_2
import cv2

file_path_1 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-01/gt/gt.txt"
file_path_2 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-02/gt/gt.txt"
file_path_3 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-03/gt/gt.txt"
file_path_4 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-05/gt/gt.txt"

paths = [file_path_1, file_path_2, file_path_3, file_path_4]

test_dict,gt = test_data_loader(paths[0])

FPS = 25
dt = 1/FPS

def phy_based_prediction(track_seq):
    
    top_left_x = physics_based_prediction_2(track_seq[:,0],dt)
    top_left_y = physics_based_prediction_2(track_seq[:,1],dt)
    width = physics_based_prediction_2(track_seq[:,2],dt)
    height = physics_based_prediction_2(track_seq[:,3],dt)

    return [top_left_x,top_left_y,width,height]

def object_tracking_fnct(test_dict,gt):
    """
    input
        test_dict : key is frame number and value is the list of [x,y,w,h] (i.e [[x_1,y_1,w_1,h_1],[x_2,y_2,w_2,h_2]])
    """

    """
    output
        output_dict : key is frame number and value is the list of [track_id,x,y,w,h]
    """

    output_dict = {}
    track_memory = {}

    for frame, objects in test_dict.items():
        current_frame_tracks = []

        now_gt = gt[frame]
        k = -1
        for obj in objects:
            k += 1
            found_track = False
            for track_id, track_seq in track_memory.items():
                    prediction = phy_based_prediction(np.array(track_seq))
                    if np.linalg.norm(np.array(prediction[:2]) - np.array(obj[:2])) < 30:  # Threshold for matching
                        track_seq.append(obj)
                        if len(track_seq) > 3:
                            track_seq.pop(0)  # Maintain only the last 3 bounding boxes
                        current_frame_tracks.append([track_id] + list(obj))
                        found_track = True
                        break

            if not found_track:
                #new_track_id = len(track_memory) + 1
                new_track_id = now_gt[k][0]
                track_memory[new_track_id] = [obj]
                current_frame_tracks.append([new_track_id] + list(obj))

        #print(current_frame_tracks)
        output_dict[frame] = current_frame_tracks

        print(f"done: {frame}")

        #if frame == 100:
        #    return output_dict
    return output_dict


def compare_dictionaries(gt, output_dict):
    """
    Compares the ground truth dictionary with the predicted output dictionary.
    Prints the differences in terms of the object IDs for bounding boxes that are exactly the same.

    Args:
        gt (dict): Ground truth dictionary with track IDs and bounding boxes.
        output_dict (dict): Predicted dictionary with track IDs and bounding boxes.
    """
    for frame in gt:
        gt_objects = gt.get(frame, [])
        predicted_objects = output_dict.get(frame, [])
        
        print(f"Frame {frame}:")
        
        # Create dictionaries for easier comparison by bounding box coordinates
        gt_dict = {tuple(obj[1:]): obj[0] for obj in gt_objects}
        predicted_dict = {tuple(obj[1:]): obj[0] for obj in predicted_objects}

        total_gt_boxes = len(gt_objects)
        matching_boxes = 0
        
        for bbox, gt_id in gt_dict.items():
            predicted_id = predicted_dict.get(bbox, None)
            if predicted_id is not None:
            #    print(f"  Bounding Box: {bbox}")
            #    print(f"    Ground Truth ID: {gt_id}")
            #    print(f"    Predicted ID: {predicted_id}")
                if gt_id == predicted_id:
            #        print("    IDs match")
                    matching_boxes += 1
            #    else:
            #        print("    IDs do not match")
            #else:
            #    print(f"  Bounding Box: {bbox}")
            #    print(f"    Ground Truth ID: {gt_id}")
            #    print(f"    Predicted: Not Available")
        
        #for bbox, predicted_id in predicted_dict.items():
            #if bbox not in gt_dict:
            #    print(f"  Bounding Box: {bbox}")
            #    print(f"    Ground Truth: Not Available")
            #    print(f"    Predicted ID: {predicted_id}")
        
        if total_gt_boxes > 0:
            matching_probability = matching_boxes / total_gt_boxes
        else:
            matching_probability = 0.0

        print(f"  Matching Probability frame {frame}: {matching_probability:.2f}")
        
        #if frame == 100:
        #    return None

# Example usage

output_dict = object_tracking_fnct(test_dict,gt)

compare_dictionaries(gt, output_dict)
def draw_tracking_info(video_path, output_dict, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set target frame size
    target_width, target_height = 1920, 1080
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    frame_idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the target size
        frame = cv2.resize(frame, (target_width, target_height))

        # Draw the bounding boxes and IDs on the frame
        if frame_idx in output_dict:
            for obj in output_dict[frame_idx]:
                track_id, left_top_x, left_top_y, width, height = obj
                right_bottom_x = left_top_x + width
                right_bottom_y = left_top_y + height

                # Draw the bounding box
                cv2.rectangle(frame, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), (0, 255, 0), 2)
                
                # Write the ID above the bounding box
                cv2.putText(frame, str(track_id), (left_top_x, left_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame into the output video
        out.write(frame)
        frame_idx += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_path = 'MOT20-01-raw.webm'  # Path to the original WEBM video
output_video_path = 'output_video_with_tracking_phy.avi'  # Path to save the new video

draw_tracking_info(video_path, output_dict, output_video_path)

