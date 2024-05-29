import torch
import torch.nn as nn
import numpy as np
from iou_cal import calculate_iou
from data_loader import organize_data
import itertools

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

def compute_iou(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)
    return calculate_iou(pred, gt)

def inference(tracking_dict, agent):
    all_iou_scores = []
    for object_id, boxes in tracking_dict.items():
        if len(boxes) < 4:
            continue  # Not enough boxes to predict subsequent states
        state = get_state(boxes, 0)  # Initial state
        predicted_boxes = []
        ground_truth_boxes = boxes[3:]  # Remaining boxes are the ground truth for comparison

        for t in range(1, len(boxes) - 2):
            action = agent.select_action(state)  # Select the best action based on the current state
            predicted_box = boxes[t + 2] + action  # Apply action to the last known box
            predicted_boxes.append(predicted_box)
            state = get_state(boxes, t)  # Update state to next time step

        iou_scores = [compute_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
        average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        all_iou_scores.append(average_iou)
        print(f"Object ID: {object_id}, Average IoU: {average_iou}")

    overall_average_iou = sum(all_iou_scores) / len(all_iou_scores) if all_iou_scores else 0
    print(f"Overall Average IoU: {overall_average_iou}")
    return overall_average_iou

# Example use
test_path = "./MOT20Labels/train/MOT20-02/gt/gt.txt"

test_tracking_dict = organize_data(test_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the agent and load the trained model
agent = Agent(12, 4, device)
agent.load_model("dqn_tracking_model.pth")

# Perform inference
overall_iou = inference(test_tracking_dict, agent)
