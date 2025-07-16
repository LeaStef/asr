#!/usr/bin/env python3
"""
Estimate model parameters for LMU ASR configurations.
"""

import math

def estimate_lmu_params(input_size, hidden_size, memory_size):
    """Estimate parameters for a single LMU layer."""
    # LMU parameters from pytorch-lmu implementation
    # Encoding vectors: e_x, e_h, e_m
    encoding_params = input_size + hidden_size + memory_size
    
    # Weight matrices: W_x, W_h, W_m
    weight_params = (hidden_size * input_size + 
                    hidden_size * hidden_size + 
                    hidden_size * memory_size)
    
    # A and B matrices (typically not learned)
    # A: memory_size x memory_size, B: memory_size x 1
    ab_params = 0  # Usually fixed, not learned
    
    return encoding_params + weight_params + ab_params

def estimate_attention_params(hidden_size, num_heads):
    """Estimate parameters for multi-head attention."""
    # Query, Key, Value, Output projections
    qkv_params = 4 * (hidden_size * hidden_size)
    
    # Layer norm parameters (weight + bias)
    layer_norm_params = 2 * hidden_size
    
    return qkv_params + layer_norm_params

def estimate_model_params(config):
    """Estimate total model parameters."""
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    memory_size = config['memory_size']
    num_lmu_layers = config['num_lmu_layers']
    vocab_size = config['vocab_size']
    use_attention = config['use_attention']
    num_attention_heads = config['num_attention_heads']
    
    total_params = 0
    
    # Input projection
    input_proj_params = input_size * hidden_size + hidden_size  # weights + bias
    total_params += input_proj_params
    
    # LMU layers
    lmu_params_per_layer = estimate_lmu_params(hidden_size, hidden_size, memory_size)
    total_lmu_params = num_lmu_layers * lmu_params_per_layer
    total_params += total_lmu_params
    
    # Attention layers
    total_attention_params = 0
    if use_attention:
        attention_params_per_layer = estimate_attention_params(hidden_size, num_attention_heads)
        total_attention_params = num_lmu_layers * attention_params_per_layer
        total_params += total_attention_params
    
    # Layer normalization (one per LMU layer)
    layer_norm_params = num_lmu_layers * 2 * hidden_size  # weight + bias per layer
    total_params += layer_norm_params
    
    # Output projection
    output_proj_params = hidden_size * hidden_size + hidden_size  # weights + bias
    total_params += output_proj_params
    
    # CTC decoder (linear layer)
    ctc_params = hidden_size * vocab_size + vocab_size  # weights + bias
    total_params += ctc_params
    
    return {
        'total_params': total_params,
        'input_projection': input_proj_params,
        'lmu_layers': total_lmu_params,
        'attention_layers': total_attention_params,
        'layer_norm': layer_norm_params,
        'output_projection': output_proj_params,
        'ctc_decoder': ctc_params,
        'lmu_params_per_layer': lmu_params_per_layer,
        'attention_params_per_layer': attention_params_per_layer if use_attention else 0
    }

def print_analysis(config_name, config, results):
    """Print detailed parameter analysis."""
    print(f"\n{config_name}:")
    print("=" * 60)
    
    print(f"Model Configuration:")
    print(f"  Input size: {config['input_size']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Memory size: {config['memory_size']}")
    print(f"  LMU layers: {config['num_lmu_layers']}")
    print(f"  Vocabulary size: {config['vocab_size']}")
    print(f"  Attention enabled: {config['use_attention']}")
    if config['use_attention']:
        print(f"  Attention heads: {config['num_attention_heads']}")
    
    print(f"\nParameter Breakdown:")
    print(f"  Input projection:     {results['input_projection']:,} params")
    print(f"  LMU layers:          {results['lmu_layers']:,} params")
    if results['attention_layers'] > 0:
        print(f"  Attention layers:    {results['attention_layers']:,} params")
    print(f"  Layer normalization: {results['layer_norm']:,} params")
    print(f"  Output projection:   {results['output_projection']:,} params")
    print(f"  CTC decoder:         {results['ctc_decoder']:,} params")
    print(f"  {'─' * 40}")
    print(f"  Total parameters:    {results['total_params']:,} params")
    
    # Calculate model size in MB (assuming float32)
    model_size_mb = results['total_params'] * 4 / (1024 * 1024)
    print(f"  Model size (FP32):   {model_size_mb:.2f} MB")
    print(f"  Model size (FP16):   {model_size_mb/2:.2f} MB")
    
    # Per-layer analysis
    print(f"\nPer-Layer Analysis:")
    print(f"  LMU params per layer: {results['lmu_params_per_layer']:,}")
    if results['attention_params_per_layer'] > 0:
        print(f"  Attention params per layer: {results['attention_params_per_layer']:,}")
        print(f"  Total per layer: {results['lmu_params_per_layer'] + results['attention_params_per_layer']:,}")
    
    # Percentages
    print(f"\nComponent Percentages:")
    total = results['total_params']
    print(f"  LMU layers: {results['lmu_layers']/total*100:.1f}%")
    if results['attention_layers'] > 0:
        print(f"  Attention: {results['attention_layers']/total*100:.1f}%")
    print(f"  CTC decoder: {results['ctc_decoder']/total*100:.1f}%")
    print(f"  Other: {(total - results['lmu_layers'] - results['attention_layers'] - results['ctc_decoder'])/total*100:.1f}%")

def main():
    """Main analysis function."""
    print("LMU ASR Model Parameter Estimation")
    print("=" * 60)
    
    # Demo configuration (from notebook)
    demo_config = {
        'input_size': 80,
        'hidden_size': 256,
        'memory_size': 128,
        'num_lmu_layers': 2,
        'vocab_size': 29,
        'use_attention': True,
        'num_attention_heads': 8
    }
    
    # Production configuration (from base_config.yaml)
    production_config = {
        'input_size': 80,
        'hidden_size': 512,
        'memory_size': 256,
        'num_lmu_layers': 4,
        'vocab_size': 29,
        'use_attention': True,
        'num_attention_heads': 8
    }
    
    # Large production configuration
    large_config = {
        'input_size': 80,
        'hidden_size': 1024,
        'memory_size': 512,
        'num_lmu_layers': 6,
        'vocab_size': 29,
        'use_attention': True,
        'num_attention_heads': 16
    }
    
    # Analyze configurations
    demo_results = estimate_model_params(demo_config)
    production_results = estimate_model_params(production_config)
    large_results = estimate_model_params(large_config)
    
    print_analysis("Demo Configuration", demo_config, demo_results)
    print_analysis("Production Configuration", production_config, production_results)
    print_analysis("Large Production Configuration", large_config, large_results)
    
    # Comparison without attention
    production_no_attention = production_config.copy()
    production_no_attention['use_attention'] = False
    production_no_attention_results = estimate_model_params(production_no_attention)
    
    print(f"\nAttention Impact Analysis (Production Config):")
    print("=" * 60)
    with_attention = production_results['total_params']
    without_attention = production_no_attention_results['total_params']
    attention_overhead = with_attention - without_attention
    
    print(f"  With attention:    {with_attention:,} params")
    print(f"  Without attention: {without_attention:,} params")
    print(f"  Attention overhead: {attention_overhead:,} params")
    print(f"  Relative increase: {attention_overhead/without_attention*100:.1f}%")
    
    # Memory requirements for training
    print(f"\nTraining Memory Requirements (Production Config):")
    print("=" * 60)
    
    # Model parameters + gradients + optimizer states (Adam: 2x params)
    # Plus activations and temporary buffers
    base_memory = with_attention * 4  # FP32 weights
    gradient_memory = with_attention * 4  # FP32 gradients
    optimizer_memory = with_attention * 8  # Adam: momentum + variance
    
    training_memory_gb = (base_memory + gradient_memory + optimizer_memory) / (1024**3)
    
    print(f"  Model weights (FP32): {base_memory / (1024**3):.2f} GB")
    print(f"  Gradients (FP32): {gradient_memory / (1024**3):.2f} GB")
    print(f"  Optimizer states: {optimizer_memory / (1024**3):.2f} GB")
    print(f"  Total (minimum): {training_memory_gb:.2f} GB")
    print(f"  With activations: ~{training_memory_gb * 2:.2f} GB")
    
    # Mixed precision training
    mixed_precision_memory = (with_attention * 2 + with_attention * 4 + with_attention * 8) / (1024**3)
    print(f"  Mixed precision: ~{mixed_precision_memory:.2f} GB")
    
    print(f"\nRecommended GPU Memory:")
    print(f"  Demo model: ≥4 GB")
    print(f"  Production model: ≥8 GB")
    print(f"  Large model: ≥16 GB")

if __name__ == "__main__":
    main()