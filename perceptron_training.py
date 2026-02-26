# trains model to guess digits (MNIST)

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ----------------------------
# Model (LeNet-ish)
# ----------------------------
class Perceptron(nn.Module):
    """LeNet-style CNN classifier for MNIST (10 classes)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits (no softmax)


# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_random_mask(x: torch.Tensor) -> torch.Tensor:
    """
    Random binary mask (0/1) same shape as x.
    Keeps ~50% pixels on average.
    Works on CPU/GPU because it uses x.device.
    """
    mask = torch.randint(0, 2, size=x.shape, device=x.device, dtype=x.dtype)
    return x * mask


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    random_masking: bool = False,
) -> Tuple[float, float]:
    """
    Returns: (mean_loss, accuracy)
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if random_masking:
            x = apply_random_mask(x)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return loss_sum / total, correct / total


# ----------------------------
# Training
# ----------------------------
def train_perceptron(
    epochs: int = 3,
    random_masking: bool = False,
    saving: bool = True,
    seed: int = 42,
    batch_size: int = 32,
    lr: float = 1e-2,
    data_root: str = "data",
    save_path: str = "data/trained_LeNet.pth",
    mask_in_validation: bool = False,
) -> Perceptron:
    """
    Trains the classifier on MNIST and optionally applies random pixel masking during training.
    Saves the best model by validation loss.
    """
    print("Training perceptron...")
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    # IMPORTANT: set download=False if you already have MNIST locally
    dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)

    # Reproducible split
    train_ds, valid_ds = random_split(
        dataset,
        [50000, 10000],
        generator=torch.Generator().manual_seed(seed),
    )

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)  # no shuffle for val

    model = Perceptron().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_valid_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            if random_masking:
                x = apply_random_mask(x)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.size(0)
            train_seen += int(x.size(0))

        train_loss = train_loss_sum / train_seen
        valid_loss, valid_acc = evaluate(
            model,
            validloader,
            criterion,
            device,
            random_masking=mask_in_validation and random_masking,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | valid_acc={valid_acc*100:.2f}%"
        )

        if saving and valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model -> {save_path}")

    return model


if __name__ == "__main__":
    mdl = train_perceptron(epochs=3, random_masking=False, saving=True)