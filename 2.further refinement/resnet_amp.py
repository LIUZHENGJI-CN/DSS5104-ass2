# ResNet-50 full fine-tuning without augmentation, 8 epochs, with AMP

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
# train_one_epoch function (AMP version)
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
# evaluate function (AMP version)
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

    # 1. dataset info
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    print("Dataset info:")
    print(info)

    # 2. device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 3. transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform_basic = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])


    # 4. dataset
    train_dataset_basic = PathMNIST(split='train', transform=train_transform_basic, download=True)
    val_dataset = PathMNIST(split='val', transform=test_transform, download=True)
    test_dataset = PathMNIST(split='test', transform=test_transform, download=True)

    print("Train:", len(train_dataset_basic))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    # 5. DataLoader
    batch_size = 64
    num_workers = 4

    train_loader_basic = DataLoader(
        train_dataset_basic,
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
    # ResNet-50 full fine-tuning with AMP
    # ==============================
    model_resnet_full = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model_resnet_full.parameters():
        param.requires_grad = True

    num_features = model_resnet_full.fc.in_features
    model_resnet_full.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    model_resnet_full = model_resnet_full.to(device)

    print(model_resnet_full.fc)

    criterion_resnet_full = nn.CrossEntropyLoss()

    optimizer_resnet_full = torch.optim.Adam(
        model_resnet_full.parameters(),
        lr=1e-5
    )

    num_epochs = 8

    # early stopping settings
    patience = 2
    min_delta = 1e-4
    best_val_loss = float("inf")
    best_epoch = 0
    counter = 0
    best_model_path = "resnet50_amp.pth"

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_params = sum(p.numel() for p in model_resnet_full.parameters())
    trainable_params = sum(p.numel() for p in model_resnet_full.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # training history
    train_losses_resnet_full, train_accs_resnet_full = [], []
    val_losses_resnet_full, val_accs_resnet_full = [], []

    # total training time tracking
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        
        train_loss, train_acc = train_one_epoch(
            model_resnet_full,
            train_loader_basic,
            criterion_resnet_full,
            optimizer_resnet_full,
            device,
            scaler,
            epoch_idx=epoch,
            num_epochs=num_epochs,
            log_interval=100
        )

        val_loss, val_acc = evaluate(
            model_resnet_full,
            val_loader,
            criterion_resnet_full,
            device
        )

        train_losses_resnet_full.append(train_loss)
        train_accs_resnet_full.append(train_acc)
        val_losses_resnet_full.append(val_loss)
        val_accs_resnet_full.append(val_acc)

        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - total_start_time

        print("-" * 60)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds | {epoch_time/60:.2f} min")
        print(f"Total elapsed: {total_elapsed:.2f} seconds | {total_elapsed/60:.2f} min")

        # early stopping check based on val loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model_resnet_full.state_dict(), best_model_path)
            print(f"Validation loss improved. Best model saved to {best_model_path}")
            print(f"Current best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        else:
            counter += 1
            print(f"No significant improvement in val loss for {counter} epoch(s).")

        if counter >= patience:
            print("=" * 60)
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    total_time = time.time() - total_start_time
    print("=" * 60)
    print("Training finished.")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # reload best model
    if best_epoch > 0:
        model_resnet_full.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Best model reloaded from {best_model_path}")

    # plot curves
    epochs = range(1, len(train_losses_resnet_full) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_resnet_full, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses_resnet_full, marker='o', label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (ResNet50 Full Fine-Tuning, AMP)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs_resnet_full, marker='o', label='Train Acc')
    plt.plot(epochs, val_accs_resnet_full, marker='o', label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy (ResNet50 Full Fine-Tuning, AMP)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # final save (best model)
    final_model_path = "resnet50_amp.pth"
    torch.save(model_resnet_full.state_dict(), final_model_path) 
    print(f"Final model saved as {final_model_path}")



if __name__ == "__main__":
    main()