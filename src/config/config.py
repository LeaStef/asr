from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig


@dataclass
class ModelConfig:
    input_size: int = 80
    hidden_size: int = 512
    memory_size: int = 256
    num_lmu_layers: int = 4
    theta: float = 1000.0
    dropout: float = 0.1
    use_fft_lmu: bool = False
    vocab_size: int = 31
    use_attention: bool = True
    num_attention_heads: int = 8
    use_downsampling: bool = False
    downsample_factor: int = 2


@dataclass
class DataConfig:
    dataset: str = "gigaspeech"
    subset: str = "xs"
    save_dir: str = "./data"
    sample_rate: int = 16000
    n_mels: int = 80
    max_seq_len: int = 1000
    augment: bool = True
    num_workers: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 16
    lr: float = 1e-3
    max_epochs: int = 50
    patience: int = 10
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    accumulate_grad_batches: int = 1


@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    sync_batchnorm: bool = True


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    distributed: Optional[DistributedConfig] = None


def create_config_from_dict(cfg_dict: DictConfig) -> Config:
    """Create Config object from OmegaConf DictConfig"""
    model_cfg = ModelConfig(**cfg_dict.model.encoder, vocab_size=cfg_dict.model.decoder.vocab_size)
    data_cfg = DataConfig(**cfg_dict.data)
    training_cfg = TrainingConfig(**cfg_dict.training)
    
    distributed_cfg = None
    if hasattr(cfg_dict, 'distributed') and cfg_dict.distributed is not None:
        distributed_cfg = DistributedConfig(**cfg_dict.distributed)
    
    return Config(
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
        distributed=distributed_cfg
    )