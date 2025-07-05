import random
from collections import deque
import torch
import numpy as np

class ReplayBuffer:
    
    def __init__(self, capacity, batch_size, device):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device
        
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = [np.array(s).squeeze(-1) for s in states]
        # print(f'len states: {len(states)}')
        # for i in range(len(states)):
        #     print(f'state dim: {states[i].squeeze(-1).shape}')
        #     #print(f'state dim: {states[i].shape}')
        states = np.stack(states, axis=0)
        next_states = [np.array(s).squeeze(-1) for s in next_states]
        next_states = np.stack(next_states, axis=0)
        return (
            torch.from_numpy(states).float().to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(rewards, dtype=torch.float).to(self.device),
            torch.from_numpy(next_states).float().to(self.device),
            torch.tensor(dones, dtype=torch.float).to(self.device)
        )
    
    def __len__(self):
        return len(self.buffer)