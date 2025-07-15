import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pytorch-lmu', 'src'))

import torch
import torch.nn as nn
from typing import Tuple, Optional
from lmu import LMU, LMUFFT


class LMUEncoder(nn.Module):
    """
    LMU-based encoder for ASR system.
    
    Takes mel-spectrogram features as input and produces encoded representations
    using a stack of LMU layers for temporal modeling.
    """
    
    def __init__(self, 
                 input_size: int = 80,
                 hidden_size: int = 512,
                 memory_size: int = 256,
                 num_lmu_layers: int = 4,
                 theta: float = 128,
                 dropout: float = 0.1,
                 use_fft_lmu: bool = False,
                 seq_len: Optional[int] = None):
        """
        Initialize LMU encoder.
        
        Args:
            input_size: Size of input features (mel-spectrogram dims)
            hidden_size: Hidden size for LMU layers
            memory_size: Memory size for LMU layers
            num_lmu_layers: Number of LMU layers to stack
            theta: LMU theta parameter (memory timescale)
            dropout: Dropout rate
            use_fft_lmu: Whether to use FFT-based LMU (requires fixed seq_len)
            seq_len: Sequence length (required for FFT-based LMU)
        """
        super(LMUEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_lmu_layers = num_lmu_layers
        self.theta = theta
        self.dropout = dropout
        self.use_fft_lmu = use_fft_lmu
        self.seq_len = seq_len
        
        if use_fft_lmu and seq_len is None:
            raise ValueError("seq_len must be specified when using FFT-based LMU")
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Stack of LMU layers
        self.lmu_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_lmu_layers):
            if use_fft_lmu:
                # FFT-based LMU (faster for fixed sequence length)
                lmu_layer = LMUFFT(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    memory_size=memory_size,
                    seq_len=seq_len,
                    theta=theta
                )
            else:
                # Standard LMU (flexible sequence length)
                lmu_layer = LMU(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    memory_size=memory_size,
                    theta=theta
                )
            
            self.lmu_layers.append(lmu_layer)
            self.layer_norms.append(nn.LayerNorm(hidden_size))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through LMU encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for masking
            
        Returns:
            encoded: Encoded features of shape (batch_size, seq_len, hidden_size)
            memory_states: List of memory states from each LMU layer
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        memory_states = []
        
        # Pass through LMU layers
        for i, (lmu_layer, layer_norm) in enumerate(zip(self.lmu_layers, self.layer_norms)):
            if self.use_fft_lmu:
                # FFT-based LMU returns (output, final_hidden)
                x, h_n = lmu_layer(x)
                memory_states.append(h_n)
            else:
                # Standard LMU returns (output, (final_hidden, final_memory))
                x, (h_n, m_n) = lmu_layer(x)
                memory_states.append((h_n, m_n))
            
            # Apply layer norm and dropout
            x = layer_norm(x)
            x = self.dropout_layer(x)
        
        # Output projection
        encoded = self.output_projection(x)
        
        return encoded, memory_states
    
    def get_output_dim(self) -> int:
        """Get output dimension of the encoder."""
        return self.hidden_size


class DownsamplingLMUEncoder(LMUEncoder):
    """
    LMU encoder with downsampling for memory efficiency.
    Reduces sequence length by a factor while maintaining temporal modeling.
    """
    
    def __init__(self, 
                 input_size: int = 80,
                 hidden_size: int = 512,
                 memory_size: int = 256,
                 num_lmu_layers: int = 4,
                 theta: float = 128,
                 dropout: float = 0.1,
                 use_fft_lmu: bool = False,
                 seq_len: Optional[int] = None,
                 downsample_factor: int = 2):
        """
        Initialize downsampling LMU encoder.
        
        Args:
            downsample_factor: Factor by which to downsample sequence length
            Other args same as LMUEncoder
        """
        super(DownsamplingLMUEncoder, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            num_lmu_layers=num_lmu_layers,
            theta=theta,
            dropout=dropout,
            use_fft_lmu=use_fft_lmu,
            seq_len=seq_len
        )
        
        self.downsample_factor = downsample_factor
        
        # Downsampling layers after every few LMU layers
        self.downsample_layers = nn.ModuleList()
        for i in range(0, num_lmu_layers, 2):  # Downsample every 2 layers
            self.downsample_layers.append(
                nn.Conv1d(hidden_size, hidden_size, 
                         kernel_size=downsample_factor, 
                         stride=downsample_factor)
            )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with downsampling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for masking
            
        Returns:
            encoded: Encoded features with reduced sequence length
            memory_states: List of memory states from each LMU layer
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        memory_states = []
        downsample_idx = 0
        
        # Pass through LMU layers with downsampling
        for i, (lmu_layer, layer_norm) in enumerate(zip(self.lmu_layers, self.layer_norms)):
            if self.use_fft_lmu:
                # FFT-based LMU returns (output, final_hidden)
                x, h_n = lmu_layer(x)
                memory_states.append(h_n)
            else:
                # Standard LMU returns (output, (final_hidden, final_memory))
                x, (h_n, m_n) = lmu_layer(x)
                memory_states.append((h_n, m_n))
            
            # Apply layer norm and dropout
            x = layer_norm(x)
            x = self.dropout_layer(x)
            
            # Apply downsampling every 2 layers
            if i % 2 == 1 and downsample_idx < len(self.downsample_layers):
                # Transpose for conv1d: (batch_size, hidden_size, seq_len)
                x = x.transpose(1, 2)
                x = self.downsample_layers[downsample_idx](x)
                x = x.transpose(1, 2)  # Back to (batch_size, seq_len, hidden_size)
                downsample_idx += 1
                
                # Update lengths if provided
                if lengths is not None:
                    lengths = lengths // self.downsample_factor
        
        # Output projection
        encoded = self.output_projection(x)
        
        return encoded, memory_states