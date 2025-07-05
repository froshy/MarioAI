import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.envs import make_env
from src.replay_buffer import ReplayBuffer
from src.neural import MNet

class Trainer:
    
    def __init__(self, cfg):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.env = make_env(cfg.env_name, cfg.stack_size)
        self.replay = ReplayBuffer(cfg.buffer_size, cfg.batch_size, self.device)
        
        # networks
        self.policy_net = MNet(cfg.stack_size, self.env.action_space.n).to(self.device)
        self.target_net = MNet(cfg.stack_size, self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # optimizer & loss
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        
        # bookkeeping
        self.batch_size = cfg.batch_size
        self.start_learning = cfg.start_learning
        self.update_every = cfg.update_every
        self.target_update_freq = cfg.target_update_freq
        self.max_frames = cfg.get("max_frames", None)
        
        # logging
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.frame_idx = 0
        self.episode_rewards = []
        
        
    def select_action(self, state, epsilon):
        if torch.rand(1).item() < epsilon:
            return self.env.action_space.sample()
        
        frame_stack = np.array(state)
        state_t = torch.from_numpy(frame_stack).float()
        state_t = state_t.squeeze(-1)
        state_t = state_t.unsqueeze(0).to(self.device)
        
        
        with torch.no_grad():
            qvals = self.policy_net(state_t)
        return qvals.argmax(1).item()
    
    
    def optimize(self):
        states, actions, rewards, next_states, dones = self.replay.sample()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(current_q, target_q)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    def train(self):
        state = self.env.reset()
        episode_reward = 0
        while self.max_frames is None or self.frame_idx < self.max_frames:
            epsilon = max(0.01, 0.1 - 0.01*(self.frame_idx/200000))
            action = self.select_action(state, epsilon)
            next_state, reward, done, info_ = self.env.step(action)
            self.replay.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            self.frame_idx += 1
            
            # learn
            if self.frame_idx > self.start_learning and self.frame_idx % self.update_every == 0:
                loss = self.optimize()
                self.writer.add_scalar('loss', loss, self.frame_idx)
                
            # target update
            if self.frame_idx % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if done:
                self.writer.add_scalar("episode_reward", episode_reward, self.frame_idx)
                state = self.env.reset()
                episode_reward = 0
            
        self.writer.close()