
import os
import hydra
import torch
import glob
from omegaconf import DictConfig, OmegaConf
from dataset.common import PuzzleDatasetMetadata
from pretrain import (
    load_synced_config,
    create_dataloader,
    create_evaluators,
    init_train_state,
    evaluate
)

@hydra.main(config_path="config", config_name="cfg_dual_head", version_base=None)
def main(cfg: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    
    # Load config
    config = load_synced_config(cfg, rank=RANK, world_size=WORLD_SIZE)
    
    print("\n" + "="*50)
    print("      TRM EVALUATION MODE (32-digit OOD)")
    print("="*50 + "\n")
    
    # Override Data Path to 32-digit OOD
    ood_path = "data/arithmetic_32"
    if os.path.exists(ood_path):
        print(f"Loading OOD Data from: {ood_path}")
        config.data_paths_test = [ood_path]
    else:
        print(f"Warning: {ood_path} not found, using default: {config.data_paths_test}")

    # Create Loader (Test Split)
    try:
        eval_loader, eval_metadata = create_dataloader(
            config, 
            "test", 
            test_set_mode=True, 
            epochs_per_iter=1, 
            global_batch_size=config.global_batch_size, 
            rank=RANK, 
            world_size=WORLD_SIZE
        )
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return

    # Find Checkpoint
    # Default to finding the latest in outputs/ or checkpoints/
    # Logic: If cfg.checkpoint_path is set, use it. Else search.
    # The user is running from current dir.
    
    ckpt_path = None
    
    # Heuristic: Search recursively for 'last.ckpt' in 'outputs/' folder (Hydra default) 
    # or the user-specific 'checkpoints' folder.
    # But pretrain.py saves to `checkpoints/Project/RunName/Step...`
    # Let's check common locations.
    
    possible_ckpts = glob.glob("outputs/**/last.ckpt", recursive=True)
    possible_ckpts += glob.glob("checkpoints/**/*.ckpt", recursive=True)
    possible_ckpts += glob.glob("wandb/**/*last.ckpt", recursive=True)
    
    # Sort by modification time
    possible_ckpts.sort(key=os.path.getmtime, reverse=True)
    
    if len(possible_ckpts) > 0:
        ckpt_path = possible_ckpts[0]
        print(f"Found checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found! Using initialized random weights (Baseline).")
    
    if ckpt_path:
        config.resume_from = ckpt_path

    # Initialize Model & State
    train_state = init_train_state(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    # Create Evaluators
    evaluators = create_evaluators(config, eval_metadata)
    
    if not evaluators:
        print("No evaluators configured!")
        return
        
    print(f"Running {len(evaluators)} Evaluators...")
    
    # Evaluation Loop
    train_state.model.eval()
    
    metrics = evaluate(
        config, 
        train_state, 
        eval_loader, 
        eval_metadata, 
        evaluators,
        rank=RANK, 
        world_size=WORLD_SIZE,
        cpu_group=None
    )
    
    print("\n" + "="*50)
    print("      EVALUATION COMPLETE")
    print("="*50 + "\n")
    if metrics:
        def print_metrics(m, prefix=""):
            for k, v in m.items():
                if isinstance(v, dict):
                    print(f"{prefix}{k}:")
                    print_metrics(v, prefix + "  ")
                elif isinstance(v, (float, int)):
                    print(f"{prefix}{k}: {v:.4f}")
                else:
                    print(f"{prefix}{k}: {v}")
        
        print_metrics(metrics)

if __name__ == "__main__":
    main()
