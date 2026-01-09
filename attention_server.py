#!/usr/bin/env python3
"""
Flask backend server for visualizing attention maps from TinyRecursiveReasoningModel.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False
    print("Warning: flask-cors not installed. CORS may not work properly.")
import yaml
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import pydantic

# Import project modules
from pretrain import PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig, load_checkpoint, create_dataloader
from puzzle_dataset import PuzzleDatasetMetadata
from utils.functions import load_model_class

app = Flask(__name__)
if cors_available:
    CORS(app)  # Enable CORS for frontend
else:
    # Manual CORS headers
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response

# Global model state
model = None
config = None
metadata = None
vocab_map_inv = None


def tokenize_addition_string(input_str: str, dataset_mode: str = "vanilla", digits: int = 3) -> List[int]:
    """Tokenize an addition string like '123+456=' into token IDs."""
    # Vocabulary mapping (matches build_addition_dataset.py)
    # 2-11: digits 0-9
    # 12: '+'
    # 13: '='
    # 14: '<CAR>' (only for lilavati1)
    vocab_map = {str(i): i + 2 for i in range(10)}
    vocab_map['+'] = 12
    vocab_map['='] = 13
    if dataset_mode == "lilavati1":
        vocab_map['<CAR>'] = 14
    
    tokens = []
    for char in input_str:
        if char in vocab_map:
            tokens.append(vocab_map[char])
        else:
            raise ValueError(f"Unknown character '{char}' in input string")
    
    return tokens


def create_vocab_map_inv(vocab_size: int, dataset_mode: str) -> Dict[int, str]:
    """Create inverse vocabulary mapping for token IDs to strings."""
    vocab_map_inv = {i+2: str(i) for i in range(10)}
    vocab_map_inv[12] = '+'
    vocab_map_inv[13] = '='
    if dataset_mode == "lilavati1" and vocab_size >= 15:
        vocab_map_inv[14] = '<CAR>'
    vocab_map_inv[0] = 'PAD'
    vocab_map_inv[1] = 'MASK'
    return vocab_map_inv


def load_model_from_config(config_path: str, checkpoint_path: str):
    """Load model from config and checkpoint files."""
    global model, config, metadata, vocab_map_inv
    
    # Load Hydra config with defaults resolved
    # The config file uses defaults that reference other configs (e.g., arch: trm)
    # We need to use Hydra's compose API to resolve these
    
    # Get absolute paths
    abs_config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(abs_config_path)
    config_name = os.path.basename(abs_config_path).replace('.yaml', '')
    
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra and compose config
    # config_dir should point to the directory containing the config file
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        hydra_config = compose(config_name=config_name)
    
    # Convert OmegaConf to dict and then to PretrainConfig
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)
    config = PretrainConfig(**config_dict)
    
    # Handle checkpoint path: if it's a directory, find the latest step_* file
    if os.path.isdir(checkpoint_path):
        # Find all step_* files
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, 'step_*'))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in directory: {checkpoint_path}")
        # Sort by modification time and get the latest
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = checkpoint_files[0]
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Load checkpoint to determine vocab_size (checkpoint is source of truth)
    checkpoint_vocab_size = None
    checkpoint_state = None
    if os.path.isfile(checkpoint_path):
        try:
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
            # Check embedding weight size to determine vocab_size
            for key in checkpoint_state.keys():
                # Handle both compiled (_orig_mod) and uncompiled model keys
                clean_key = key.replace("_orig_mod.", "")
                if "embed_tokens.embedding_weight" in clean_key:
                    checkpoint_vocab_size = checkpoint_state[key].shape[0]
                    print(f"Detected vocab_size={checkpoint_vocab_size} from checkpoint embedding")
                    break
                # Also check lm_head
                if "lm_head.weight" in clean_key:
                    checkpoint_vocab_size = checkpoint_state[key].shape[0]
                    print(f"Detected vocab_size={checkpoint_vocab_size} from checkpoint lm_head")
                    break
        except Exception as e:
            print(f"Warning: Could not determine vocab_size from checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    if checkpoint_vocab_size is None:
        raise ValueError(f"Could not determine vocab_size from checkpoint: {checkpoint_path}")
    
    # Update dataset_mode based on vocab_size from checkpoint
    if checkpoint_vocab_size >= 15:
        config.dataset_mode = "lilavati1"
        print(f"Detected dataset_mode=lilavati1 from vocab_size={checkpoint_vocab_size}")
    else:
        config.dataset_mode = "vanilla"
        print(f"Detected dataset_mode=vanilla from vocab_size={checkpoint_vocab_size}")
    
    config.load_checkpoint = checkpoint_path
    
    # Load dataset metadata to get seq_len and other info
    dataset_path = config.data_paths[0]
    metadata_path = os.path.join(dataset_path, "train", "dataset.json")
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    metadata = PuzzleDatasetMetadata(**metadata_dict)
    
    # Warn if vocab_size mismatch
    if metadata.vocab_size != checkpoint_vocab_size:
        print(f"Warning: Dataset vocab_size ({metadata.vocab_size}) differs from checkpoint vocab_size ({checkpoint_vocab_size}). Using checkpoint vocab_size.")
    
    # Create vocab mapping using checkpoint vocab_size
    vocab_map_inv = create_vocab_map_inv(checkpoint_vocab_size, config.dataset_mode)
    
    # Create model config using checkpoint vocab_size
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=1,  # Single example for visualization
        vocab_size=checkpoint_vocab_size,  # Use vocab_size from checkpoint
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )
    
    # Instantiate model
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    with torch.device("cuda"):
        loaded_model = model_cls(model_cfg)
        loaded_model = loss_head_cls(loaded_model, **config.arch.loss.__pydantic_extra__)
        
        # Load checkpoint - handle compiled model state dict
        if config.load_checkpoint is not None:
            print(f"Loading checkpoint {config.load_checkpoint}")
            state_dict = torch.load(config.load_checkpoint, map_location="cuda")
            
            # Strip _orig_mod. prefix if present (from compiled models)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("_orig_mod."):
                    new_key = key[len("_orig_mod."):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Handle puzzle embedding shape mismatch if needed
            puzzle_emb_name = "model.inner.puzzle_emb.weights"
            if puzzle_emb_name in new_state_dict:
                expected_shape = loaded_model.model.puzzle_emb.weights.shape
                puzzle_emb = new_state_dict[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                    new_state_dict[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous().cuda()
                    )
            
            loaded_model.load_state_dict(new_state_dict, assign=True)
        
        # Move model to CUDA explicitly
        loaded_model = loaded_model.cuda()
        loaded_model.eval()
    
    return loaded_model, config, metadata


def extract_attention_weights(model: nn.Module, batch: Dict[str, torch.Tensor], 
                              layer_indices: Optional[List[int]] = None,
                              head_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights from the model by temporarily modifying Block forward methods.
    Returns a dictionary mapping layer indices to attention weight tensors.
    """
    attention_weights_cuda = {}  # Store on CUDA during forward pass
    
    # Find all attention layers in the model
    # The model structure: model.model.inner.L_level.layers[i].self_attn
    inner_model = model.model.inner
    l_level = inner_model.L_level
    
    # Store original forward methods
    original_block_forwards = {}
    original_attn_forwards = {}
    
    # Temporarily modify Block forward methods to capture attention weights
    if layer_indices is None:
        layer_indices = list(range(len(l_level.layers)))
    
    for layer_idx in layer_indices:
        if layer_idx < len(l_level.layers):
            layer = l_level.layers[layer_idx]
            if hasattr(layer, 'self_attn') and layer.self_attn is not None:
                # Store original methods
                original_block_forwards[layer_idx] = layer.forward
                original_attn_forwards[layer_idx] = layer.self_attn.forward
                
                # Create modified block forward that captures attention weights
                def make_modified_block_forward(block_layer, attn_module, idx, orig_forward):
                    def modified_forward(cos_sin, hidden_states):
                        # Call attention with return_attn_weights=True
                        attn_result = attn_module.forward(cos_sin, hidden_states, return_attn_weights=True)
                        if isinstance(attn_result, tuple):
                            attn_output, attn_weights = attn_result
                            # Store attention weights on CUDA first (detach to avoid gradients)
                            # We'll move to CPU after the forward pass is complete
                            attention_weights_cuda[idx] = attn_weights.detach()
                        else:
                            attn_output = attn_result
                        
                        # Continue with rest of block forward (same as original but using attn_output)
                        # attn_output should already be on the same device as hidden_states (CUDA)
                        from models.layers import rms_norm
                        if block_layer.config.mlp_t:
                            # This layer uses MLP, not attention, so use original forward
                            return orig_forward(cos_sin, hidden_states)
                        else:
                            # Ensure attn_output is on the same device as hidden_states
                            # This should already be the case, but double-check
                            if attn_output.device != hidden_states.device:
                                attn_output = attn_output.to(hidden_states.device)
                            hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=block_layer.norm_eps)
                            out = block_layer.mlp(hidden_states)
                            hidden_states = rms_norm(hidden_states + out, variance_epsilon=block_layer.norm_eps)
                            return hidden_states
                    return modified_forward
                
                layer.forward = make_modified_block_forward(layer, layer.self_attn, layer_idx, original_block_forwards[layer_idx])
    
    # Run forward pass
    try:
        with torch.no_grad():
            # Ensure all batch tensors are on CUDA
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Use CUDA device context (like in pretrain.py)
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
                # Ensure carry tensors are on CUDA
                if hasattr(carry, 'inner_carry'):
                    if hasattr(carry.inner_carry, 'z_H'):
                        carry.inner_carry.z_H = carry.inner_carry.z_H.cuda()
                    if hasattr(carry.inner_carry, 'z_L'):
                        carry.inner_carry.z_L = carry.inner_carry.z_L.cuda()
                
                # Run inference for one step (this will call inner model forward)
                # The inner model forward runs H_cycles and L_cycles, and attention is in L_level layers
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=set()
                )
    finally:
        # Restore original forwards
        for layer_idx in layer_indices:
            if layer_idx < len(l_level.layers):
                layer = l_level.layers[layer_idx]
                if layer_idx in original_block_forwards:
                    layer.forward = original_block_forwards[layer_idx]
                if layer_idx in original_attn_forwards:
                    layer.self_attn.forward = original_attn_forwards[layer_idx]
        
        # Move attention weights to CPU after forward pass is complete
        # Convert to float32 first (BFloat16 is not supported by numpy)
        attention_weights = {k: v.to(torch.float32).cpu() for k, v in attention_weights_cuda.items()}
    
    return attention_weights


@app.route('/api/attention', methods=['POST'])
def get_attention():
    """API endpoint to compute and return attention maps."""
    global model, config, metadata, vocab_map_inv
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please load model first.'}), 400
    
    try:
        data = request.json
        input_str = data.get('input', '')
        layer_indices = data.get('layers', None)  # List of layer indices, None = all
        head_indices = data.get('heads', None)    # List of head indices, None = all
        
        if not input_str:
            return jsonify({'error': 'Input string is required'}), 400
        
        # Tokenize input
        tokens = tokenize_addition_string(input_str, config.dataset_mode, config.digits)
        
        # Pad to seq_len
        seq_len = metadata.seq_len
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        else:
            tokens = tokens + [0] * (seq_len - len(tokens))  # Pad with 0 (PAD token)
        
        # Create batch (model expects inputs, puzzle_identifiers, and labels for current_data)
        # For visualization, we don't need actual labels, but the model structure expects them
        seq_len = metadata.seq_len
        batch = {
            'inputs': torch.tensor([tokens], dtype=torch.int32).cuda(),
            'puzzle_identifiers': torch.tensor([0], dtype=torch.int32).cuda(),  # Default puzzle ID
            'labels': torch.full((1, seq_len), -100, dtype=torch.int32).cuda(),  # IGNORE_LABEL_ID for all positions
        }
        
        # Extract attention weights
        attention_weights = extract_attention_weights(
            model, batch, 
            layer_indices=layer_indices,
            head_indices=head_indices
        )
        
        # Convert to JSON-serializable format
        result = {
            'input_tokens': tokens,
            'input_string': input_str,
            'token_labels': [vocab_map_inv.get(t, '?') for t in tokens],
            'attention_maps': {}
        }
        
        # Process attention weights
        for layer_idx, attn_weights in attention_weights.items():
            # attn_weights: [B, H, S, S]
            attn_weights = attn_weights[0]  # Remove batch dimension: [H, S, S]
            
            # Filter heads if specified
            if head_indices is not None:
                attn_weights = attn_weights[head_indices]
            else:
                attn_weights = attn_weights  # All heads
            
            # Convert to list and filter by head
            layer_data = {}
            num_heads = attn_weights.shape[0]
            for head_idx in range(num_heads):
                if head_indices is None or head_idx in head_indices:
                    # Convert to float32 first (BFloat16 is not supported by numpy)
                    head_weights = attn_weights[head_idx].to(torch.float32).numpy().tolist()  # [S, S]
                    layer_data[f'head_{head_idx}'] = head_weights
            
            result['attention_maps'][f'layer_{layer_idx}'] = layer_data
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    global model, config, metadata
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    # Get number of layers
    inner_model = model.model.inner
    l_level = inner_model.L_level
    num_layers = len(l_level.layers)
    
    # Get number of heads
    num_heads = config.arch.num_heads
    
    return jsonify({
        'num_layers': num_layers,
        'num_heads': num_heads,
        'seq_len': metadata.seq_len,
        'vocab_size': metadata.vocab_size,
        'dataset_mode': config.dataset_mode if hasattr(config, 'dataset_mode') else 'unknown',
        'digits': config.digits if hasattr(config, 'digits') else None
    })


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load a model from config and checkpoint files."""
    global model, config, metadata, vocab_map_inv
    
    try:
        data = request.json
        config_path = data.get('config_path')
        checkpoint_path = data.get('checkpoint_path')
        
        if not config_path or not checkpoint_path:
            return jsonify({'error': 'config_path and checkpoint_path are required'}), 400
        
        if not os.path.exists(config_path):
            return jsonify({'error': f'Config file not found: {config_path}'}), 400
        
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': f'Checkpoint file not found: {checkpoint_path}'}), 400
        
        model, config, metadata = load_model_from_config(config_path, checkpoint_path)
        vocab_map_inv = create_vocab_map_inv(metadata.vocab_size, config.dataset_mode)
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_info': {
                'num_layers': len(model.model.inner.L_level.layers),
                'num_heads': config.arch.num_heads,
                'seq_len': metadata.seq_len,
                'vocab_size': metadata.vocab_size
            }
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - serves the HTML frontend or returns server info."""
    # Try to serve the HTML file if it exists
    html_path = os.path.join(os.path.dirname(__file__), 'attention_viewer.html')
    if os.path.exists(html_path):
        return send_from_directory(os.path.dirname(__file__), 'attention_viewer.html')
    else:
        # Return JSON info if HTML file not found
        return jsonify({
            'status': 'ok',
            'service': 'Attention Map Visualization Server',
            'model_loaded': model is not None,
            'endpoints': {
                'health': '/health',
                'model_info': '/api/model_info',
                'load_model': '/api/load_model (POST)',
                'attention': '/api/attention (POST)'
            },
            'frontend': 'Open attention_viewer.html in your browser and set server URL to this server'
        })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


def main():
    parser = argparse.ArgumentParser(
        description='Attention visualization server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Start server on default host/port
  python attention_server.py --config config/pretrain_addition_vanilla_d3.yaml --checkpoint checkpoints/trm-lilavati/vanilla_trm_d3

  # Start server on specific IP and port
  python attention_server.py --config config/pretrain_addition_vanilla_d3.yaml --checkpoint checkpoints/trm-lilavati/vanilla_trm_d3 --host 0.0.0.0 --port 8080

  # Start server without loading model at startup
  python attention_server.py --host 0.0.0.0 --port 5000 --no-load
        '''
    )
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint file (or directory containing step_* files)')
    parser.add_argument('--host', type=str, default='127.0.0.1', 
                       help='Host/IP to bind to (default: 127.0.0.1). Use 0.0.0.0 to accept connections from any IP.')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--no-load', action='store_true', help='Do not load model at startup')
    
    args = parser.parse_args()
    
    # Load model if provided
    if not args.no_load and args.config and args.checkpoint:
        print(f"Loading model from config: {args.config}, checkpoint: {args.checkpoint}")
        try:
            load_model_from_config(args.config, args.checkpoint)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Model not loaded at startup. Use /api/load_model endpoint to load a model.")
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Access the frontend at: attention_viewer.html?server=http://{args.host}:{args.port}")
    if args.host == '127.0.0.1' or args.host == 'localhost':
        print(f"Or open: attention_viewer.html?server=http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
