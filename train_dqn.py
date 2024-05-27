import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from iou_cal import calculate_iou
from data_loader import organize_data
import itertools


path = "./MOT20Labels/train/MOT20-01/gt/gt.txt"

class DQNNetwork(nn.Module):
    __slots__ = 'fc1', 'fc2', 'fc3',

    def __init__(self, state_dim: int, action_dim: int):
        super(DQNNetwork, self).__init__()

        # Input is 12 elements from 3 previous boxes (state) + Δx, Δy, Δw, Δh (action)
        self.fc1: nn.Linear = nn.Linear(state_dim + action_dim, state_dim)
        self.fc2: nn.Linear = nn.Linear(state_dim, action_dim)
        self.fc3: nn.Linear = nn.Linear(action_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    __slots__ = 'buffer',

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Agent:
    __slots__ = 'state_dim', 'n_actions', 'replay_buffer', 'model', 'optimizer', 'criterion', 'action_candidates',

    def __init__(self, state_dim: int, n_actions: int, buffer: ReplayBuffer):
        self.state_dim: int = state_dim
        self.n_actions: int = n_actions
        self.replay_buffer: ReplayBuffer = buffer
        self.model: DQNNetwork = DQNNetwork(state_dim, n_actions).to(device)
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters())
        self.criterion: nn.MSELoss = nn.MSELoss()

        self.action_candidates: torch.Tensor = torch.tensor(
            list(itertools.product(range(-4, 5), repeat=4)),
            dtype=torch.float32
        ).to(device)

    def select_action(self, state: np.ndarray, epsilon=0) -> np.ndarray:
        # print("select_action")
        if np.random.random() < epsilon:
            return np.random.randint(-30, 30, 4)  # Random action
        else:
            states_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            expanded_states = states_tensor.repeat(self.action_candidates.size(0), 1)
            model_input = torch.cat((expanded_states, self.action_candidates), dim=1)

            with torch.no_grad():
                q_values = self.model(model_input).squeeze()

            best_action_idx = torch.argmax(q_values)
            return self.action_candidates[best_action_idx].cpu().numpy()

    def save_model(self, filepath: str):
        # Save model state_dict and optimizer state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def update_policy(self, batch_size: int):
        #print("update policy")
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
        batch_action = torch.tensor(batch_action, dtype=torch.float32).to(device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(device)
        batch_done = torch.tensor(batch_done, dtype=torch.bool).to(device)

        # Calculate the current Q-values using the main network
        state_action_pairs = torch.cat((batch_state, batch_action), dim=1)
        state_action_values = self.model(state_action_pairs)

        # Calculate the target Q-values using the target network
        with torch.no_grad():
            num_candidates = self.action_candidates.shape[0]
            expanded_next_states = batch_next_state.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, batch_next_state.size(-1))

            repeated_actions = self.action_candidates.repeat(batch_next_state.size(0), 1)
            model_input = torch.cat((expanded_next_states, repeated_actions), dim=1)
            target_next_q_values = self.model(model_input).view(batch_next_state.size(0), num_candidates)

            next_action_values, _ = target_next_q_values.max(dim=1)
            expected_state_action_values = batch_reward + (1 - batch_done.float()) * 0.99 * next_action_values

        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_state(boxes, index) -> np.ndarray:
    # Flattens the previous 3 boxes into a single state vector
    return np.concatenate(boxes[index:index + 3]).ravel()


def compute_iou(box_t2, action, box_t3) -> float:
    # Calculate IoU between predicted and ground truth boxes
    # This is a placeholder; you should implement this correctly
    box_t2 = np.array(box_t2)
    action = np.array(action)
    box_t3 = np.array(box_t3)
    return calculate_iou(box_t2 + action, box_t3)


def train(tracking: dict, agent: Agent, num_epochs: int, batch_size: int, save_path: str):
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_decay = 500
    epsilon_by_epoch = lambda frame_idx: epsilon_end + (epsilon_start - epsilon_end) * np.exp(
        -1. * frame_idx / epsilon_decay)

    for epoch in range(num_epochs):
        entire_reward = []
        for object_id, boxes in tracking.items():
            cnt = 0
            if len(boxes) < 4:
                continue  # Skip if there are less than 4 boxes
            state = get_state(boxes, 0)  # Initial state
            total_reward = 0
            for t in range(1, len(boxes) - 3):
                cnt += 1
                action = agent.select_action(state, epsilon_by_epoch(epoch))
                next_state = np.concatenate((boxes[t + 1], boxes[t + 2], boxes[t + 2] + action)).ravel()
                reward = compute_iou(boxes[t + 2], action, boxes[t + 3])
                done = t == len(boxes) - 4
                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            # print(f"Epoch: {epoch}, Object ID: {object_id}, Total Reward: {total_reward / cnt}")
            entire_reward.append(total_reward / cnt)
        print(f"epoch: {epoch}, reward:{np.average(np.array(entire_reward))}")
        for i in range(100):
            if len(agent.replay_buffer) > batch_size:
                agent.update_policy(batch_size)
                # print(f"epoch: {epoch}, update_idx: {i}")
        # print("Agent update policy complete")
        # Save model at the end of each epoch or periodically
        if (epoch + 1) % 10 == 0:
            agent.save_model(save_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device.type}")
tracking_dict = organize_data(path)
replay_buffer = ReplayBuffer(1000)
agent = Agent(12, 4, replay_buffer)
train(tracking_dict, agent, 1000, 32, "dqn_tracking_model.pth")
