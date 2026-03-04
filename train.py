import os
import time
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.config import ModelConfig, TrainingConfig
from src.transformer import VisionTransformer
from src.data import get_cifar100_loaders, Mixup, CutMix


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0
):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    config,
    device,
    epoch,
    mixup_fn=None,
    cutmix_fn=None,
):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        use_mixup = mixup_fn is not None and random.random() < 0.5
        use_cutmix = cutmix_fn is not None and random.random() < 0.5 and not use_mixup

        if use_mixup:
            images, targets_a, targets_b, lam = mixup_fn(images, targets)
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix_fn(images, targets)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        optimizer.zero_grad()

        with autocast(enabled=config.use_amp):
            logits, aux_loss = model(images)

            if use_mixup or use_cutmix:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(
                    logits, targets_b
                )
            else:
                loss = criterion(logits, targets)

            if aux_loss is not None:
                total_loss = loss + config.aux_loss_coef * aux_loss
            else:
                total_loss = loss

        scaler.scale(total_loss).backward()

        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        (acc1,) = accuracy(logits, targets, topk=(1,))

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
        if aux_loss is not None:
            aux_loss_meter.update(aux_loss.item(), images.size(0))

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "aux_loss": aux_loss_meter.avg if aux_loss_meter.count > 0 else 0,
        "lr": scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits, _ = model(images)
        loss = criterion(logits, targets)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))

    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg, "acc5": acc5_meter.avg}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_acc, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"], checkpoint["best_acc"]


def main():
    parser = argparse.ArgumentParser(description="Train ViT-MoE on CIFAR-100")
    parser.add_argument(
        "--config", type=str, default="base", choices=["tiny", "base", "large"]
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
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

    train_config = TrainingConfig()

    set_seed(train_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionTransformer(model_config).to(device)

    print(f"Model parameters: {model.get_num_params():,}")
    print(
        f"Model config: embed_dim={model_config.embed_dim}, layers={model_config.num_layers}"
    )
    print(
        f"MoE: {model_config.num_experts} experts, top-{model_config.num_experts_per_token}"
    )

    train_loader, test_loader = get_cifar100_loaders(
        train_config, model_config.img_size
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    total_steps = len(train_loader) * train_config.num_epochs
    warmup_steps = len(train_loader) * train_config.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=train_config.min_lr / train_config.learning_rate,
    )

    scaler = GradScaler(enabled=train_config.use_amp)

    mixup_fn = Mixup(train_config.mixup_alpha) if train_config.mixup_alpha > 0 else None
    cutmix_fn = (
        CutMix(train_config.cutmix_alpha) if train_config.cutmix_alpha > 0 else None
    )

    start_epoch = 0
    best_acc = 0.0

    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, device
        )
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    if args.eval_only:
        metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Test Acc@1: {metrics['acc1']:.2f}%")
        print(f"Test Acc@5: {metrics['acc5']:.2f}%")
        return

    print(f"\nStarting training for {train_config.num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, train_config.num_epochs):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            train_config,
            device,
            epoch,
            mixup_fn,
            cutmix_fn,
        )

        epoch_time = time.time() - start_time

        if (epoch + 1) % train_config.eval_every == 0:
            test_metrics = evaluate(model, test_loader, criterion, device)

            is_best = test_metrics["acc1"] > best_acc
            best_acc = max(test_metrics["acc1"], best_acc)

            print(
                f"Epoch {epoch + 1:3d}/{train_config.num_epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['acc']:.2f}% | "
                f"Test Acc@1: {test_metrics['acc1']:.2f}% | "
                f"Test Acc@5: {test_metrics['acc5']:.2f}% | "
                f"Best: {best_acc:.2f}% | "
                f"LR: {train_metrics['lr']:.6f}"
            )

            if is_best:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_acc,
                    checkpoint_dir / "best.pt",
                )
        else:
            print(
                f"Epoch {epoch + 1:3d}/{train_config.num_epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['acc']:.2f}% | "
                f"Aux Loss: {train_metrics['aux_loss']:.4f} | "
                f"LR: {train_metrics['lr']:.6f}"
            )

        if (epoch + 1) % train_config.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_acc,
                checkpoint_dir / f"epoch_{epoch + 1}.pt",
            )

    print("=" * 70)
    print(f"Training complete. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
