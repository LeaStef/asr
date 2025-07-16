import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pytorch-lmu', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from lmu import LMU, LMUFFT


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for sequence modeling.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = query.size()
        
        # Store residual connection
        residual = query
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


class SelfAttention(nn.Module):
    """
    Self-attention layer for sequence modeling.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize self-attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Self-attention output (batch_size, seq_len, d_model)
        """
        return self.attention(x, x, x, mask)


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
                 seq_len: Optional[int] = None,
                 use_attention: bool = True,
                 num_attention_heads: int = 8):
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
            use_attention: Whether to use attention layers before LMU layers
            num_attention_heads: Number of attention heads
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
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        
        if use_fft_lmu and seq_len is None:
            raise ValueError("seq_len must be specified when using FFT-based LMU")
        
        # Input projection layer
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Stack of attention and LMU layers
        self.attention_layers = nn.ModuleList()
        self.lmu_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_lmu_layers):
            # Add attention layer before each LMU layer
            if use_attention:
                attention_layer = SelfAttention(
                    d_model=hidden_size,
                    num_heads=num_attention_heads,
                    dropout=dropout
                )
                self.attention_layers.append(attention_layer)
            else:
                self.attention_layers.append(None)
            
            # Add LMU layer
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
        Forward pass through attention-enhanced LMU encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for masking
            
        Returns:
            encoded: Encoded features of shape (batch_size, seq_len, hidden_size)
            memory_states: List of memory states from each LMU layer
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # Create attention mask if lengths provided
        attention_mask = None
        if lengths is not None:
            batch_size, seq_len = x.shape[:2]
            attention_mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            for i, length in enumerate(lengths):
                attention_mask[i, :length, :length] = 1
        
        memory_states = []
        
        # Pass through attention and LMU layers
        for i, (attention_layer, lmu_layer, layer_norm) in enumerate(zip(self.attention_layers, self.lmu_layers, self.layer_norms)):
            
            # Apply attention before LMU if enabled
            if self.use_attention and attention_layer is not None:
                x = attention_layer(x, attention_mask)
            
            # Apply LMU layer
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
                 use_attention: bool = True,
                 num_attention_heads: int = 8,
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
            seq_len=seq_len,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads
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
        Forward pass with downsampling and attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for masking
            
        Returns:
            encoded: Encoded features with reduced sequence length
            memory_states: List of memory states from each LMU layer
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)
        
        # Create attention mask if lengths provided
        attention_mask = None
        if lengths is not None:
            batch_size, seq_len = x.shape[:2]
            attention_mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            for i, length in enumerate(lengths):
                attention_mask[i, :length, :length] = 1
        
        memory_states = []
        downsample_idx = 0
        
        # Pass through attention and LMU layers with downsampling
        for i, (attention_layer, lmu_layer, layer_norm) in enumerate(zip(self.attention_layers, self.lmu_layers, self.layer_norms)):
            
            # Apply attention before LMU if enabled
            if self.use_attention and attention_layer is not None:
                x = attention_layer(x, attention_mask)
            
            # Apply LMU layer
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
                
                # Update attention mask for new sequence length
                if attention_mask is not None:
                    new_seq_len = x.shape[1]
                    batch_size = x.shape[0]
                    attention_mask = torch.zeros(batch_size, new_seq_len, new_seq_len, device=x.device)
                    if lengths is not None:
                        for j, length in enumerate(lengths):
                            attention_mask[j, :length, :length] = 1
        
        # Output projection
        encoded = self.output_projection(x)
        
        return encoded, memory_states