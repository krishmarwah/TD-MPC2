import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

import warnings
warnings.filterwarnings('ignore')
import torch
import hydra
from termcolor import colored

# Your custom ENV
from envArj import RobotEnv

# TD-MPC2 infrastructure
from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True

@hydra.main(config_path='.', config_name='config', version_base=None)
def train(cfg: dict):
    
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0
    
    # Parse hydra config
    cfg = parse_cfg(cfg)

    # Multi-GPU selection
    # FIX: Use getattr instead of 'in' operator as cfg is a class instance, not a dict
    if getattr(cfg, 'cuda_id', None) is not None:
        torch.cuda.set_device(cfg.cuda_id)
        print(colored(f"[GPU] Using cuda:{cfg.cuda_id}", "green", attrs=["bold"]))
    else:
        print(colored("[GPU] Using default CUDA device", "yellow"))

    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer

    # Initialize Environment
    env = RobotEnv(gui=False)
    
    # Initialize Agent
    agent = TDMPC2(cfg)
    
    # Initialize Buffer
    buffer = Buffer(cfg)
    
    # Initialize Logger
    logger = Logger(cfg)

    # Create trainer
    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger
    )

    print("Starting training...")
    trainer.train()
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    train()
