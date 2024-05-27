from data_loader import test_data_loader
import numpy as np
import torch
import torch.nn as nn
import itertools
import cv2

file_path_1 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-01/gt/gt.txt"
file_path_2 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-02/gt/gt.txt"
file_path_3 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-03/gt/gt.txt"
file_path_4 = "D:/DGIST/4학년/Reinforcement Learning/Project/Object_tracking_Main_10frame/MOT20Labels/train/MOT20-05/gt/gt.txt"

paths = [file_path_1, file_path_2, file_path_3, file_path_4]

test_dict,gt = test_data_loader(paths[0])

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, state_dim)
        self.fc2 = nn.Linear(state_dim, action_dim)
        self.fc3 = nn.Linear(action_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_dim, n_actions, device):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device
        self.model = DQNNetwork(state_dim, n_actions).to(device)
        self.action_candidates = torch.tensor(
            list(itertools.product(range(-4, 5), repeat=4)),        
            dtype=torch.float32
        ).to(device)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        expanded_states = state_tensor.repeat(self.action_candidates.size(0), 1)
        model_input = torch.cat((expanded_states, self.action_candidates), dim=1)

        with torch.no_grad():
            q_values = self.model(model_input).squeeze()

        best_action_idx = torch.argmax(q_values)
        return self.action_candidates[best_action_idx].cpu().numpy()

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode

def get_state(boxes, index):
    return np.concatenate(boxes[index:index+3]).ravel()

def DQN_based_prediction(input_seq, agent):
    np_seq = np.array(input_seq) 
    if len(np_seq) == 1:
        return input_seq[0]
    if len(np_seq) == 2:
        return input_seq[1]
    if len(np_seq) == 3:
        state = get_state(np_seq, 0)
        action = agent.select_action(state)
        predicted_box = np_seq[2] + action
        return predicted_box

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
                    prediction = DQN_based_prediction(track_seq, agent)
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

        #if frame == 4:
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
        
        #if frame == 4:
        #    return None

# Example usage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(12, 4, device)
agent.load_model("dqn_tracking_model.pth")

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


video_path = 'MOT20-01-raw.WEBM'  # Path to the original WEBM video
output_video_path = 'output_video_with_tracking_dqn.avi'  # Path to save the new video

draw_tracking_info(video_path, output_dict, output_video_path)

