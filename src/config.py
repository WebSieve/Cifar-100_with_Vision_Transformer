from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6

    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1

    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_token: int = 2
    num_shared_experts: int = 1
    aux_loss_coef: float = 0.01

    use_mla: bool = True
    kv_compression_ratio: int = 4
    qk_rope_dim: int = 16

    use_rms_norm: bool = True
    norm_eps: float = 1e-6

    use_cls_token: bool = True
    use_pos_embed: bool = True

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def intermediate_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def kv_lora_rank(self) -> int:
        return self.embed_dim // self.kv_compression_ratio

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def seq_len(self) -> int:
        return self.num_patches + (1 if self.use_cls_token else 0)


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-5

    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0

    grad_clip: float = 1.0

    use_amp: bool = True

    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    save_every: int = 10
    eval_every: int = 1

    data_dir: str = "~/Downloads/datasets/cifar-100"
    checkpoint_dir: str = "./checkpoints"
