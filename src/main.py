import os
import sys
import signal

import hydra
from omegaconf import DictConfig
from src.trainer import Trainer
from src.utils import set_seed, ensure_dir, save_checkpoint

@hydra.main(config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.checkpoint_dir)
    
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    
    def _on_interrupt(signum, frame):
        print('\n received interrupt, saving checkpoint...')
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, f'INTERRUPT_{trainer.frame_idx}.pt'),
            trainer.policy_net,
            trainer.optimizer,
            trainer.frame_idx,
            best_reward=getattr(trainer, 'best_reward', None)
        )
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_interrupt)
    
    trainer.train()


if __name__ == "__main__":
    main()