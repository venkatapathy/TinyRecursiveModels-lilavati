
import os
import hydra
import torch
import glob
import copy
from omegaconf import DictConfig, OmegaConf
from dataset.common import PuzzleDatasetMetadata
from pretrain import (
    load_synced_config,
    create_dataloader,
    create_evaluators,
    init_train_state,
    evaluate,
    TrainState
)
import wandb
import sys

# Hack to allow unpickling TrainState which was saved as __main__.TrainState in pretrain.py
sys.modules['__main__'].TrainState = TrainState

@hydra.main(config_path="config", config_name="cfg_dual_head", version_base=None)
def main(cfg: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    
    # Load config
    config = load_synced_config(cfg, rank=RANK, world_size=WORLD_SIZE)
    
    print("\n" + "="*50)
    print("      TRM EVALUATION MODE (32-digit OOD)")
    print("="*50 + "\n")
    
    # Override Data Path based on split
    # Default behavior: use OOD (test) path if split is not specified or "test"
    # If split is "val", use ID (val) path (which is same as train path usually, but we want val set)
    
    # Check for split argument in overrides
    split_arg = "test"
    for override in cfg.get("overrides", []):
        if override.startswith("split="):
            split_arg = override.split("=")[1]
            
    # Also check cfg itself if added to config
    if "split" in cfg:
        split_arg = cfg.split

    print(f"Evaluation Split: {split_arg.upper()}")
    
    # Logic:
    # If split="val": usage ID data (data_paths). Loader will load "val" split.
    # If split="test": usage OOD data (data_paths_test). Loader will load "test" split.
    
    target_data_paths = config.data_paths
    dataset_split = "val"
    
    if split_arg == "test":
        target_data_paths = config.data_paths_test
        dataset_split = "test" 
    elif split_arg == "val":
        target_data_paths = config.data_paths # ID data
        dataset_split = "val"
        # Ensure we are pointing to the right place for ID val
        # Usually data_paths points to the folder containing train/val/test for ID
    
    # Update config to use the selected paths
    # We cheat a bit: we set data_paths_test to the target paths so create_dataloader(..., split="test") works generic
    # OR we just call create_dataloader with the correct split and paths manually.
    
    # Let's verify paths exist
    final_paths = []
    for p in target_data_paths:
        if os.path.exists(p):
            final_paths.append(p)
        else:
            print(f"Warning: Path {p} not found.")
            
    if not final_paths:
        print("No valid data paths found!")
        return

    print(f"Loading Data from: {final_paths}")

    # Create Loader
    # We invoke create_dataloader. Note: pretrain.py's create_dataloader logic for "test" split uses data_paths_test.
    # We want to force it to use our `final_paths`.
    # We can perform a temporary patch on config.
    
    config_for_loader = copy.deepcopy(config)
    config_for_loader.data_paths_test = final_paths
    
    try:
        # We always verify against the requested split ("test" or "val")
        # But `create_dataloader` with split="test" uses `data_paths_test`.
        # If we want to evaluate on "val" split of ID data, we should pass split="val" 
        # AND ensure the loader uses the correct paths.
        # pretrain.py:109: dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths
        
        # So:
        # if split="test": loader uses config.data_paths_test (which we set to final_paths)
        # if split="val": loader uses config.data_paths. We should set config.data_paths = final_paths to be sure.
        
        config_for_loader.data_paths = final_paths
        
        eval_loader, eval_metadata = create_dataloader(
            config_for_loader, 
            dataset_split, 
            test_set_mode=True, 
            epochs_per_iter=1, 
            global_batch_size=config.global_batch_size, 
            rank=RANK, 
            world_size=WORLD_SIZE
        )
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return

    # Find Checkpoint ... (rest is same, but we search for checkpoint only if not provided)
    ckpt_path = config.load_checkpoint
    
    if not ckpt_path:
        # Search priority:
        # 0. User specified checkpoint folder via +checkpoint_folder=...
        # 1. Specific project checkpoint dir (checkpoints/trm-arthmetic-icml/<run_name>)
        # 2. General output/checkpoints dirs
        
        # Check for user-provided checkpoint folder override
        checkpoint_folder = None
        for override in cfg.get("overrides", []):
            if override.startswith("checkpoint_folder="):
                checkpoint_folder = override.split("=")[1]
        
        if "checkpoint_folder" in cfg: # Also check if it's in the config object
            checkpoint_folder = cfg.checkpoint_folder

        possible_ckpts = []
        
        if checkpoint_folder:
             print(f"Searching in user-specified folder: {checkpoint_folder}")
             patterns = [f"{checkpoint_folder}/step_*", f"{checkpoint_folder}/*.ckpt"]
             for pattern in patterns:
                 possible_ckpts.extend(glob.glob(pattern, recursive=True))
        
        # If no checkpoint found in user-specified folder (or not specified), search in run-specific folder
        if not possible_ckpts and config.run_name:
            print(f"Searching for checkpoints matching run_name: {config.run_name}")
            run_patterns = [
                f"checkpoints/trm-arthmetic-icml/{config.run_name}/step_*",
                f"checkpoints/trm-arthmetic-icml/{config.run_name}/*.ckpt"
            ]
            for pattern in run_patterns:
                possible_ckpts.extend(glob.glob(pattern, recursive=True))
                
        # If still no checkpoints, search globally (fallback)
        if not possible_ckpts:
            print("No run-specific checkpoints found. Searching globally...")
            fallback_patterns = [
                "outputs/**/last.ckpt",
                "checkpoints/**/*.ckpt",
                "checkpoints/**/step_*",
                "wandb/**/*last.ckpt"
            ]
            for pattern in fallback_patterns:
                possible_ckpts.extend(glob.glob(pattern, recursive=True))

        # Filter out 'evaluator_' directories which are just logs/results
        possible_ckpts = [p for p in possible_ckpts if "evaluator_" not in p and not os.path.isdir(p) or "step_" in os.path.basename(p)]
        
        # If user specified a folder, strictly filter for that folder (already done by search but for safety)
        if checkpoint_folder:
             # Normalize path to handle trailing slashes etc.
             norm_folder = os.path.normpath(checkpoint_folder)
             possible_ckpts = [p for p in possible_ckpts if norm_folder in os.path.normpath(p)]
        
        # Sort by modification time
        possible_ckpts.sort(key=os.path.getmtime, reverse=True)
        
        if len(possible_ckpts) > 0:
            ckpt_path = possible_ckpts[0]
            print(f"Found checkpoint: {ckpt_path}")
        else:
            search_patterns = [f"checkpoints/trm-arthmetic-icml/{config.run_name}/step_*"] if config.run_name else ["checkpoints/**/step_*"]
            raise ValueError(f"No checkpoint found! Searched patterns: {search_patterns}. Please specify +load_checkpoint=/path/to/ckpt")
    
    if ckpt_path:
        config.load_checkpoint = ckpt_path

    # Check for compile flag in overrides
    compile_arg = True
    for override in cfg.get("overrides", []):
        if override.startswith("compile="):
            compile_arg = override.split("=")[1].lower() == "true"
    if "compile" in cfg:
        compile_arg = cfg.compile
    
    if not compile_arg:
        print("Disabling torch.compile via DISABLE_COMPILE environment variable.")
        os.environ["DISABLE_COMPILE"] = "1"
    
    # SYSTEMIC FIX: Ensure vocab_size in config matches metadata exactly to avoid side-effects in init_train_state
    if hasattr(config.arch, "__pydantic_extra__"):
        config.arch.__pydantic_extra__["vocab_size"] = eval_metadata.vocab_size
        print(f"Overriding config.arch.vocab_size to {eval_metadata.vocab_size} to match metadata.")

    # Initialize Model & State
    train_state = init_train_state(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    # Create Evaluators - pass the updated config so they know which paths to use if they look at it
    # But usually evaluators just use metadata.
    # Note: create_evaluators might prefer `data_paths` or `data_paths_test`.
    # Let's pass the modified config.
    evaluators = create_evaluators(config_for_loader, eval_metadata)
    
    if not evaluators:
        print("No evaluators configured!")
        return
        
    # Initialize WandB
    if config.project_name:
        wandb_config = config.model_dump()
        wandb_config.update({
            "model": "TRM",
            "variant": config.dataset_mode,
            "digits": config.digits,
            "eval_only": True,
            "split": split_arg
        })
        run_name = config.run_name
        if not run_name:
            run_name = f"eval_{config.dataset_mode}_{split_arg}"
        else:
            run_name = f"{run_name}_{split_arg}"
            
        wandb.init(
            project=config.project_name, 
            name=run_name, 
            config=wandb_config,
            settings=wandb.Settings(_disable_stats=True)
        )

    print(f"Running {len(evaluators)} Evaluators on {dataset_split.upper()} split...")
    
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
        
        # Log to WandB
        if config.project_name:
            wandb_metrics = {}
            for dataset, dataset_metrics in metrics.items():
                prefix = f"{dataset}_{split_arg}" # Distinguish val vs test in logs
                if isinstance(dataset_metrics, dict):
                    for k, v in dataset_metrics.items():
                        wandb_metrics[f"{prefix}/{k}"] = v
                else:
                    wandb_metrics[prefix] = dataset_metrics
            
            wandb.log(wandb_metrics)
            wandb.finish()
            
        # Save metrics to JSON locally for table generation
        os.makedirs("results", exist_ok=True)
        # Use run_name for filename, ensure it's safe
        json_filename = f"{run_name}.json".replace("/", "_").replace(" ", "_")
        json_path = os.path.join("results", json_filename)
        
        import json
        import numpy as np

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        # We need to flatten the metrics if they are nested like {dataset: {metric: val}}
        # Similar to wandb logic
        flat_metrics = {}
        for dataset, dataset_metrics in metrics.items():
            prefix = f"{dataset}_{split_arg}"
            if isinstance(dataset_metrics, dict):
                for k, v in dataset_metrics.items():
                    flat_metrics[f"{prefix}/{k}"] = v
            else:
                flat_metrics[prefix] = dataset_metrics
                
        with open(json_path, "w") as f:
            json.dump(flat_metrics, f, indent=4, cls=NumpyEncoder)

        print(f"Saved results to {json_path}")

if __name__ == "__main__":
    main()
