import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from iou_cal import calculate_iou
from data_loader import organize_data
import itertools

path = "C:/seokmin_personal/Object_tracking_Main/MOT20Labels/train/MOT20-01/gt/gt.txt"

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, state_dim)  # Input is 12 elements from 3 previous boxes (state) + Δx, Δy, Δw, Δh (action)
        self.fc2 = nn.Linear(state_dim, action_dim)
        self.fc3 = nn.Linear(action_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, n_actions, replay_buffer, target_update_freq=10):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.replay_buffer = replay_buffer
        self.model = DQNNetwork(state_dim, n_actions)
        self.target_model = DQNNetwork(state_dim, n_actions)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.target_update_freq = target_update_freq
        self.update_count = 0

        # Initialize the target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def select_action(self, state, epsilon=0):
        if random.random() < epsilon:
            return np.array([random.randint(-4, 4) for _ in range(4)])  # Random action
        else:
            value_range = range(-4, 5)
            action_candidates = np.array(list(itertools.product(value_range, repeat=4)))

            with torch.no_grad():
                action_q_values = [
                    float(self.model(torch.tensor(np.concatenate((state, action)).ravel(), dtype=torch.float32)).data)
                    for action in action_candidates
                ]
                return action_candidates[np.argmax(action_q_values)]

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)

    def update_policy(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        transitions = self.replay_buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.tensor(batch_state, dtype=torch.float32)
        batch_action = torch.tensor(batch_action, dtype=torch.float32)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
        batch_done = torch.tensor(batch_done, dtype=torch.bool)

        # Calculate the current Q-values using the main network
        state_action_values = torch.cat([
            self.model(torch.cat((s, a), dim=0).unsqueeze(0))
            for s, a in zip(batch_state, batch_action)
        ])

        # Calculate the target Q-values using Double DQN
        with torch.no_grad():
            next_action_candidates = np.array(list(itertools.product(range(-4, 5), repeat=4)))
            model_next_q_values = torch.cat([
                self.model(torch.cat((next_s, torch.tensor(a, dtype=torch.float32)), dim=0).unsqueeze(0))
                for next_s in batch_next_state for a in next_action_candidates
            ])
            model_next_q_values = model_next_q_values.view(batch_size, -1)
            best_next_actions = model_next_q_values.argmax(dim=1)

            target_next_q_values = torch.cat([
                self.target_model(torch.cat((next_s, torch.tensor(next_action_candidates[a_idx], dtype=torch.float32)), dim=0).unsqueeze(0))
                for next_s, a_idx in zip(batch_next_state, best_next_actions)
            ])
            next_action_values = target_next_q_values.view(-1)
            expected_state_action_values = batch_reward + (1 - batch_done.float()) * 0.99 * next_action_values

        # Compute the loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def get_state(boxes, index):
    return np.concatenate(boxes[index:index + 3]).ravel()

def compute_iou(box_t2, action, box_t3):
    box_t2 = np.array(box_t2)
    action = np.array(action)
    box_t3 = np.array(box_t3)
    return calculate_iou(box_t2 + action, box_t3)

def train(tracking_dict, agent, num_epochs, batch_size, save_path):
    epsilon_start = 1

    epsilon_end = 0.01
    epsilon_decay = 500
    epsilon_by_epoch = lambda frame_idx: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * frame_idx / epsilon_decay)

    for epoch in range(num_epochs):
        entire_reward = []
        for object_id, boxes in tracking_dict.items():
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
            print(f"Epoch: {epoch}, Object ID: {object_id}, Total Reward: {total_reward / cnt}")
            entire_reward.append(total_reward / cnt)
        print(f"epoch: {epoch}, reward: {np.average(np.array(entire_reward))}")
        for i in range(100):
            if len(agent.replay_buffer) > batch_size:
                agent.update_policy(batch_size)
                print(f"epoch: {epoch}, update_idx: {i}")
        if (epoch + 1) % 10 == 0:
            agent.save_model(save_path)

tracking_dict = organize_data(path)
replay_buffer = ReplayBuffer(1000)
agent = Agent(12, 4, replay_buffer)
train(tracking_dict, agent, 10000, 32, "dqn_tracking_model.pth")
