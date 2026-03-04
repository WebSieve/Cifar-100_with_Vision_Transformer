import sys
import torch


def print_model_info():
    from src.config import ModelConfig
    from src.transformer import VisionTransformer

    print("=" * 60)
    print("Vision Transformer with Mixture of Experts for CIFAR-100")
    print("=" * 60)

    configs = {
        "tiny": ModelConfig(embed_dim=192, num_heads=4, num_layers=4, num_experts=4),
        "base": ModelConfig(),
        "large": ModelConfig(embed_dim=384, num_heads=12, num_layers=8, num_experts=16),
    }

    print("\nAvailable model configurations:\n")

    for name, config in configs.items():
        model = VisionTransformer(config)
        params = model.get_num_params()

        print(f"  {name.upper()}:")
        print(f"    Embedding dim:     {config.embed_dim}")
        print(f"    Layers:            {config.num_layers}")
        print(f"    Attention heads:   {config.num_heads}")
        print(
            f"    Experts:           {config.num_experts} (top-{config.num_experts_per_token})"
        )
        print(f"    Parameters:        {params:,}")
        print()

    print("Usage:")
    print("  Train:    python train.py --config base")
    print(
        "  Evaluate: python evaluate.py --checkpoint checkpoints/best.pt --config base"
    )
    print()

    print("Testing forward pass with BASE config...")
    config = configs["base"]
    model = VisionTransformer(config)

    x = torch.randn(2, 3, 32, 32)
    logits, aux_loss = model(x)

    print(f"  Input:     {x.shape}")
    print(f"  Output:    {logits.shape}")
    print(
        f"  Aux loss:  {aux_loss.item():.4f}"
        if aux_loss is not None
        else "  Aux loss:  None"
    )
    print()
    print("All systems operational.")
    print("=" * 60)


def main():
    print_model_info()


if __name__ == "__main__":
    main()
