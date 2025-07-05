import random
import os
from datetime import datetime
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def save_checkpoint(path, model, optimizer, frame_idx, best_reward=None):
    torch.save({
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'frame_idx': frame_idx,
        'best_reward': best_reward,
    }, path)

def load_checkpoint(path, model, optimizer=None):
    data = torch.load(path, map_location='cpu')
    model.load_state_dict(data['model_state'])
    if optimizer:
        optimizer.load_state_dict(data['opt_state'])
    return data.get('frame_idx', 0), data.get('best_reward', None)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    
def make_log_dir(base_dir):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    full = os.path.join(base_dir, timestamp)
    ensure_dir(full)
    return full