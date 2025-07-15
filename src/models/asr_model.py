import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pytorch-lmu', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
from .lmu_encoder import LMUEncoder, DownsamplingLMUEncoder


class CTCDecoder(nn.Module):
    """
    CTC decoder for character-level speech recognition.
    """
    
    def __init__(self, input_size: int, vocab_size: int):
        """
        Initialize CTC decoder.
        
        Args:
            input_size: Size of input features from encoder
            vocab_size: Size of vocabulary (including CTC blank token)
        """
        super(CTCDecoder, self).__init__()
        
        self.input_size = input_size
        self.vocab_size = vocab_size
        
        # Linear layer for CTC output
        self.ctc_projection = nn.Linear(input_size, vocab_size)
        
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CTC decoder.
        
        Args:
            encoder_output: Encoded features of shape (batch_size, seq_len, input_size)
            
        Returns:
            log_probs: Log probabilities of shape (batch_size, seq_len, vocab_size)
        """
        # Project to vocabulary size
        logits = self.ctc_projection(encoder_output)  # (batch_size, seq_len, vocab_size)
        
        # Apply log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs


class LMUASRModel(nn.Module):
    """
    Complete LMU-based ASR model with CTC decoder.
    """
    
    def __init__(self, config):
        """
        Initialize LMU ASR model.
        
        Args:
            config: Configuration object with model parameters
        """
        super(LMUASRModel, self).__init__()
        
        self.config = config
        
        # Create encoder
        if getattr(config, 'use_downsampling', False):
            self.encoder = DownsamplingLMUEncoder(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                num_lmu_layers=config.num_lmu_layers,
                theta=config.theta,
                dropout=config.dropout,
                use_fft_lmu=config.use_fft_lmu,
                seq_len=getattr(config, 'seq_len', None),
                downsample_factor=getattr(config, 'downsample_factor', 2)
            )
        else:
            self.encoder = LMUEncoder(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                num_lmu_layers=config.num_lmu_layers,
                theta=config.theta,
                dropout=config.dropout,
                use_fft_lmu=config.use_fft_lmu,
                seq_len=getattr(config, 'seq_len', None)
            )
        
        # Create CTC decoder
        self.decoder = CTCDecoder(
            input_size=self.encoder.get_output_dim(),
            vocab_size=config.vocab_size
        )
        
        # CTC loss function
        self.ctc_loss = nn.CTCLoss(blank=config.vocab_size - 1, reduction='mean', zero_infinity=True)
        
    def forward(self, spectrograms: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through ASR model.
        
        Args:
            spectrograms: Mel spectrograms of shape (batch_size, seq_len, n_mels)
            input_lengths: Optional sequence lengths for masking
            
        Returns:
            log_probs: Log probabilities of shape (batch_size, seq_len, vocab_size)
            memory_states: List of memory states from encoder
        """
        # Encode spectrograms
        encoded, memory_states = self.encoder(spectrograms, input_lengths)
        
        # Decode to vocabulary
        log_probs = self.decoder(encoded)
        
        return log_probs, memory_states
    
    def compute_loss(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                    input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            log_probs: Log probabilities of shape (batch_size, seq_len, vocab_size)
            targets: Target sequences of shape (batch_size, max_target_len)
            input_lengths: Input sequence lengths
            target_lengths: Target sequence lengths
            
        Returns:
            loss: CTC loss value
        """
        # CTC loss expects (seq_len, batch_size, vocab_size)
        log_probs = log_probs.transpose(0, 1)
        
        # Flatten targets for CTC loss
        targets = targets.view(-1)
        
        # Compute CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss
    
    def decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor, 
              beam_width: int = 1) -> List[List[int]]:
        """
        Decode log probabilities to sequences.
        
        Args:
            log_probs: Log probabilities of shape (batch_size, seq_len, vocab_size)
            input_lengths: Input sequence lengths
            beam_width: Beam width for beam search (1 for greedy decoding)
            
        Returns:
            decoded_sequences: List of decoded sequences for each batch item
        """
        batch_size = log_probs.size(0)
        decoded_sequences = []
        
        for i in range(batch_size):
            seq_len = input_lengths[i].item()
            seq_log_probs = log_probs[i, :seq_len]  # (seq_len, vocab_size)
            
            if beam_width == 1:
                # Greedy decoding
                decoded = self._greedy_decode(seq_log_probs)
            else:
                # Beam search decoding
                decoded = self._beam_search_decode(seq_log_probs, beam_width)
            
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def _greedy_decode(self, log_probs: torch.Tensor) -> List[int]:
        """
        Greedy CTC decoding.
        
        Args:
            log_probs: Log probabilities of shape (seq_len, vocab_size)
            
        Returns:
            decoded: Decoded sequence as list of token indices
        """
        # Get most likely tokens
        best_path = torch.argmax(log_probs, dim=-1)  # (seq_len,)
        
        # Remove blanks and consecutive duplicates
        decoded = []
        blank_token = self.config.vocab_size - 1
        
        prev_token = None
        for token in best_path:
            token = token.item()
            if token != blank_token and token != prev_token:
                decoded.append(token)
            prev_token = token
        
        return decoded
    
    def _beam_search_decode(self, log_probs: torch.Tensor, beam_width: int) -> List[int]:
        """
        Beam search CTC decoding.
        
        Args:
            log_probs: Log probabilities of shape (seq_len, vocab_size)
            beam_width: Beam width for search
            
        Returns:
            decoded: Best decoded sequence as list of token indices
        """
        seq_len, vocab_size = log_probs.shape
        blank_token = self.config.vocab_size - 1
        
        # Initialize beam with empty sequence
        beam = [{'sequence': [], 'score': 0.0, 'last_token': None}]
        
        for t in range(seq_len):
            candidates = []
            
            for beam_item in beam:
                for token in range(vocab_size):
                    score = beam_item['score'] + log_probs[t, token].item()
                    
                    if token == blank_token:
                        # Blank token - extend current sequence
                        candidates.append({
                            'sequence': beam_item['sequence'][:],
                            'score': score,
                            'last_token': token
                        })
                    elif token == beam_item['last_token']:
                        # Repeat token - don't add to sequence
                        candidates.append({
                            'sequence': beam_item['sequence'][:],
                            'score': score,
                            'last_token': token
                        })
                    else:
                        # New token - add to sequence
                        new_sequence = beam_item['sequence'][:] + [token]
                        candidates.append({
                            'sequence': new_sequence,
                            'score': score,
                            'last_token': token
                        })
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = candidates[:beam_width]
        
        # Return best sequence
        return beam[0]['sequence']
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config) -> LMUASRModel:
    """
    Create LMU ASR model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        model: LMU ASR model instance
    """
    return LMUASRModel(config)


def load_model(checkpoint_path: str, config) -> LMUASRModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        
    Returns:
        model: Loaded model instance
    """
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def apply_weight_norm(model: nn.Module) -> nn.Module:
    """
    Apply weight normalization to model layers.
    
    Args:
        model: PyTorch model
        
    Returns:
        model: Model with weight normalization applied
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module = nn.utils.weight_norm(module)
    
    return model