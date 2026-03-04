import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.config import ModelConfig, TrainingConfig
from src.transformer import VisionTransformer
from src.data import get_cifar100_loaders


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()

    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, _ = model(images)
        loss = criterion(logits, targets)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        batch_size = targets.size(0)
        total_correct_1 += acc1.item() * batch_size / 100
        total_correct_5 += acc5.item() * batch_size / 100
        total_samples += batch_size
        total_loss += loss.item() * batch_size

    return {
        "loss": total_loss / total_samples,
        "acc1": total_correct_1 / total_samples * 100,
        "acc5": total_correct_5 / total_samples * 100,
    }


@torch.no_grad()
def predict(model, image, device, class_names=None):
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    logits, _ = model(image)

    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = probs.topk(5, dim=-1)

    results = []
    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        label = class_names[idx.item()] if class_names else str(idx.item())
        results.append((label, prob.item()))

    return results


def load_model(checkpoint_path, config, device):
    model = VisionTransformer(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT-MoE on CIFAR-100")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default="base", choices=["tiny", "base", "large"]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.config == "tiny":
        model_config = ModelConfig(
            embed_dim=192, num_heads=4, num_layers=4, num_experts=4
        )
    elif args.config == "large":
        model_config = ModelConfig(
            embed_dim=384, num_heads=12, num_layers=8, num_experts=16
        )
    else:
        model_config = ModelConfig()

    train_config = TrainingConfig(batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, model_config, device)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Parameters: {model.get_num_params():,}")

    _, test_loader = get_cifar100_loaders(train_config, model_config.img_size)

    print("\nEvaluating...")
    metrics = evaluate(model, test_loader, device)

    print(f"\nResults:")
    print(f"  Test Loss:  {metrics['loss']:.4f}")
    print(f"  Test Acc@1: {metrics['acc1']:.2f}%")
    print(f"  Test Acc@5: {metrics['acc5']:.2f}%")


if __name__ == "__main__":
    main()
