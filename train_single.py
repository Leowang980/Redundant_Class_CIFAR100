"""
Single experiment: train ResNet on CIFAR-100 with num_classes (100/101/105/110).
No pretrained model. Logs to a dedicated .log file.
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import resnet18_cifar

# CIFAR-100 normalization
CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_STD = [0.2675, 0.2565, 0.2761]


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])
    return train_tf, test_tf


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def setup_logging(log_path):
    """Configure logging to file and console."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=100, choices=[100, 101, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150])
    parser.add_argument("--log_file", type=str, default="train_100.log")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64, help="64 when 4 processes share GPU")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_history", type=str, default="", help="Path to save epoch history (loss, acc) for plotting")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Unified random seed for reproducibility and fair comparison across experiments
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_logging(args.log_file)
    logging.info(f"=== Experiment: num_classes={args.num_classes}, seed={args.seed} ===")
    logging.info(f"Log file: {args.log_file}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    train_tf, test_tf = get_transforms()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)

    train_ds = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=test_tf)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = resnet18_cifar(num_classes=args.num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.2)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = Path(args.log_file).parent / f"best_num_classes_{args.num_classes}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  -> Saved best model (acc: {best_acc:.2f}%)")

    logging.info(f"Training complete. Best test accuracy: {best_acc:.2f}%")

    if args.save_history:
        import json
        with open(args.save_history, "w") as f:
            json.dump(history, f, indent=2)
        logging.info(f"History saved to {args.save_history}")


if __name__ == "__main__":
    main()
