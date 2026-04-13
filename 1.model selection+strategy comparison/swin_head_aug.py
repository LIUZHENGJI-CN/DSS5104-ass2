# This is a script for Swin-T Feature Extraction + Augmentation

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

from medmnist import INFO
from medmnist.dataset import PathMNIST


# ==============================
# define training function with AMP support
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    epoch_idx=None, num_epochs=None, log_interval=100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.time()
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - epoch_start
            current_loss = running_loss / total
            current_acc = correct / total

            if epoch_idx is not None and num_epochs is not None:
                print(
                    f"[Epoch {epoch_idx+1}/{num_epochs}] "
                    f"Batch {batch_idx+1}/{num_batches} | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f} | "
                    f"Elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)"
                )
            else:
                print(
                    f"Batch {batch_idx+1}/{num_batches} | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f} | "
                    f"Elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)"
                )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ==============================
# define evaluation function
# ==============================
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.squeeze().long().to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    print(sys.executable)

    torch.backends.cudnn.benchmark = True

    data_flag = 'pathmnist'
    info = INFO[data_flag]
    print("Dataset info:")
    print(info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # train: with augmentation
    train_transform_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # val / test: no augmentation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_dataset_aug = PathMNIST(split='train', transform=train_transform_aug, download=True)
    val_dataset = PathMNIST(split='val', transform=test_transform, download=True)
    test_dataset = PathMNIST(split='test', transform=test_transform, download=True)

    print("Train:", len(train_dataset_aug))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    batch_size = 64

    # num_workers = 4 to speed up data loading, adjust accordingly based on your system
    num_workers = 4

    train_loader_aug = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    num_classes = 9

    # ==============================
    # Swin-T Feature Extraction + Augmentation
    # freeze backbone, train custom head with augmentation
    # ==============================
    model_swin_fe_aug = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

    for param in model_swin_fe_aug.parameters():
        param.requires_grad = False

    num_features = model_swin_fe_aug.head.in_features
    model_swin_fe_aug.head = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    for param in model_swin_fe_aug.head.parameters():
        param.requires_grad = True

    model_swin_fe_aug = model_swin_fe_aug.to(device)

    print(model_swin_fe_aug.head)

    criterion_swin_fe_aug = nn.CrossEntropyLoss()

    # define optimizer and scheduler / set initial learning rate (different from CNNs)
    optimizer_swin_fe_aug = torch.optim.AdamW(
        model_swin_fe_aug.head.parameters(),
        lr=3e-4,
        weight_decay=0.01
    )

    num_epochs = 4
    scheduler_swin_fe_aug = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_swin_fe_aug,
        T_max=num_epochs,
        eta_min=6e-5
    )

    # use AMP for faster training on GPU
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_params = sum(p.numel() for p in model_swin_fe_aug.parameters())
    trainable_params = sum(p.numel() for p in model_swin_fe_aug.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_losses_swin_fe_aug, train_accs_swin_fe_aug = [], []
    val_losses_swin_fe_aug, val_accs_swin_fe_aug = [], []

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model_swin_fe_aug,
            train_loader_aug, # use augmented training data
            criterion_swin_fe_aug,
            optimizer_swin_fe_aug,
            device,
            scaler,
            epoch_idx=epoch,
            num_epochs=num_epochs,
            log_interval=100
        )

        val_loss, val_acc = evaluate(
            model_swin_fe_aug,
            val_loader,
            criterion_swin_fe_aug,
            device
        )

        train_losses_swin_fe_aug.append(train_loss)
        train_accs_swin_fe_aug.append(train_acc)
        val_losses_swin_fe_aug.append(val_loss)
        val_accs_swin_fe_aug.append(val_acc)

        scheduler_swin_fe_aug.step()
        current_lr = optimizer_swin_fe_aug.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - total_start_time

        print("-" * 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished")
        print(f"LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds | {epoch_time/60:.2f} min")
        print(f"Total elapsed: {total_elapsed:.2f} seconds | {total_elapsed/60:.2f} min")

    total_time = time.time() - total_start_time
    print("=" * 60)
    print("Training finished.")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Total training time: {total_time/60:.2f} minutes")


    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_swin_fe_aug, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses_swin_fe_aug, marker='o', label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (Swin FE + AUG)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs_swin_fe_aug, marker='o', label='Train Acc')
    plt.plot(epochs, val_accs_swin_fe_aug, marker='o', label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy (Swin FE + AUG)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    torch.save(model_swin_fe_aug.state_dict(), "swin_fe_aug_amp.pth")
    print("Model saved as swin_fe_aug_amp.pth")


if __name__ == "__main__":
    main()