# Vision Transformer with Mixture of Experts for CIFAR-100

A state-of-the-art Vision Transformer (ViT) implementation featuring Mixture of Experts (MoE) for CIFAR-100 image classification.

## Features

- **Vision Transformer (ViT)** architecture optimized for CIFAR-100
- **Multi-Head Latent Attention (MLA)** from DeepSeek for 4x KV cache compression
- **Mixture of Experts (MoE)** with load balancing for increased model capacity
- **RMSNorm** for 15% faster normalization compared to LayerNorm
- **SwiGLU** activation function in feed-forward networks
- Modern training techniques: Mixup, CutMix, label smoothing, cosine LR schedule
- Automatic Mixed Precision (AMP) training support
- Comprehensive documentation with explained versions of all modules

## Architecture Overview

```
Input Image (32×32×3)
        │
        ▼
┌───────────────────┐
│  Patch Embedding  │  4×4 patches → 64 patches + CLS token
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Position Embedding│  Learnable positional encodings
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Transformer Block │  × 6 layers
│  ├─ RMSNorm + MLA │  Multi-Head Latent Attention
│  └─ RMSNorm + MoE │  Mixture of Experts (8 experts, top-2)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   RMSNorm + Head  │  Classification (100 classes)
└───────────────────┘
```

## Project Structure

```
├── src/                    
│   ├── config.py           
│   ├── embeddings.py       
│   ├── normalization.py    
│   ├── attention.py        
│   ├── moe.py              
│   ├── transformer.py      
│   └── data.py             
│
├── explained/              
│   ├── config_explained.py
│   ├── embeddings_explained.py
│   ├── normalization_explained.py
│   ├── attention_explained.py
│   ├── moe_explained.py
│   └── transformer_explained.py
│
├── train.py               
├── evaluate.py            
├── main.py                
└── checkpoints/           
```

## Installation

```bash
# Clone the repository
git clone https://github.com/WebSieve/Cifar-100_with_Vision_Transformer.git
cd Cifar-100_with_transformers

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Training

```bash
# Basic training with default configuration
python train.py

# Training with custom parameters
python train.py --epochs 100 --batch_size 128 --lr 1e-3

# Training with specific features
python train.py --use_moe --use_mla --use_mixup --use_cutmix
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --checkpoint checkpoints/best_model.pt

# Get predictions on specific images
python evaluate.py --checkpoint checkpoints/best_model.pt --images path/to/images/
```

### Model Information

```bash
# Show model architecture and parameter count
python main.py
```

## Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 32 | Input image size |
| `patch_size` | 4 | Size of each patch |
| `embed_dim` | 256 | Embedding dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of transformer blocks |
| `mlp_ratio` | 4.0 | MLP hidden dimension ratio |
| `num_experts` | 8 | Number of experts in MoE |
| `num_experts_per_token` | 2 | Top-k experts per token |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Training batch size |
| `epochs` | 100 | Number of training epochs |
| `learning_rate` | 1e-3 | Initial learning rate |
| `weight_decay` | 0.05 | Weight decay for AdamW |
| `warmup_epochs` | 5 | Learning rate warmup epochs |
| `label_smoothing` | 0.1 | Label smoothing factor |
| `mixup_alpha` | 0.8 | Mixup interpolation strength |
| `cutmix_alpha` | 1.0 | CutMix interpolation strength |

## Key Components

### Multi-Head Latent Attention (MLA)

MLA reduces KV cache memory by 4x through low-rank projections:

- Compresses keys and values to a lower-dimensional latent space
- Reconstructs full KV representations during attention computation
- Maintains model quality while reducing memory footprint

### Mixture of Experts (MoE)

Sparse MoE increases model capacity without proportional compute increase:

- 8 experts, each a SwiGLU FFN
- Top-2 routing per token
- Load balancing auxiliary loss prevents expert collapse
- Efficient batched computation across experts

### RMSNorm

Root Mean Square Normalization provides faster normalization:

- 15% speedup over LayerNorm
- Removes mean-centering operation
- Maintains training stability

## Training Tips

1. **Learning Rate**: Start with 1e-3 for AdamW, use cosine decay
2. **Batch Size**: 128-256 works well for CIFAR-100
3. **Augmentation**: Enable Mixup and CutMix for better generalization
4. **MoE Balance**: Monitor auxiliary loss to ensure balanced expert usage
5. **Gradient Clipping**: Use max_norm=1.0 for stability

## Performance

Expected performance on CIFAR-100 test set:

| Model Variant | Parameters | Top-1 Accuracy |
|--------------|------------|----------------|
| ViT (baseline) | ~4M | ~75% |
| ViT + MoE | ~8M | ~78% |
| ViT + MLA + MoE | ~8M | ~79% |

*Results may vary based on training configuration and random seed.*

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Original ViT
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - Multi-Head Latent Attention
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Mixture of Experts
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU

## License

MIT License - see LICENSE file for details.
