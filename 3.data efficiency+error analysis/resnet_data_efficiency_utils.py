# This file contains all the helper functions for running the ResNet50 data efficiency experiments.

import os
import json
import time
import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torchvision import transforms, models

from medmnist import PathMNIST
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# =========================
# 1. Basic helpers
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================
# 2. Class names
# =========================
CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth_muscle",
    "normal_colon_mucosa",
    "cancer_associated_stroma",
    "colorectal_adenocarcinoma_epithelium"
]


# =========================
# 3. Transforms
# =========================
def get_transforms(use_train_aug: bool = True):
    if use_train_aug:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


# =========================
# 4. Load datasets
# =========================
def load_datasets(use_train_aug: bool = True):
    train_transform, eval_transform = get_transforms(use_train_aug=use_train_aug)

    train_dataset_full = PathMNIST(
        split="train",
        transform=train_transform,
        download=True
    )
    val_dataset = PathMNIST(
        split="val",
        transform=eval_transform,
        download=True
    )
    test_dataset = PathMNIST(
        split="test",
        transform=eval_transform,
        download=True
    )

    return train_dataset_full, val_dataset, test_dataset


# =========================
# 5. Stratified train subset
# =========================
def create_stratified_train_subset(train_dataset, fraction: float, seed: int):
    if fraction == 1.0:
        return Subset(train_dataset, list(range(len(train_dataset))))

    labels = np.array(train_dataset.labels).squeeze()
    indices = np.arange(len(labels))

    subset_indices, _ = train_test_split(
        indices,
        train_size=fraction,
        stratify=labels,
        random_state=seed
    )

    subset_indices = sorted(subset_indices.tolist())
    return Subset(train_dataset, subset_indices)


# =========================
# 6. Dataloaders
# =========================
def create_dataloaders(
    train_subset,
    val_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int
):
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader


# =========================
# 7. Build ResNet50 full model
#    IMPORTANT: use user's custom head
# =========================
def build_resnet50_full_model(num_classes: int = 9, pretrained: bool = True):
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = models.resnet50(weights=weights)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model


# =========================
# 8. Train / eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        if batch_idx % 100 == 0 or batch_idx == total_batches:
            current_loss = running_loss / len(all_labels)
            current_acc = accuracy_score(all_labels, all_preds)
            print(
                f"    Batch {batch_idx}/{total_batches} | "
                f"Running Loss: {current_loss:.4f} | "
                f"Running Acc: {current_acc:.4f}"
            )

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_basic(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_full_metrics(model, loader, device, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES

    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(class_names)))
    )

    per_class_df = pd.DataFrame({
        "class_idx": list(range(len(class_names))),
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    })

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_df": per_class_df,
        "confusion_matrix": cm,
        "all_labels": all_labels,
        "all_preds": all_preds
    }


def get_best_worst_classes(per_class_df: pd.DataFrame):
    sorted_df = per_class_df.sort_values("f1", ascending=False).reset_index(drop=True)
    best_class = sorted_df.iloc[0].to_dict()
    worst_class = sorted_df.iloc[-1].to_dict()
    return best_class, worst_class


# =========================
# 9. Plot confusion matrix
# =========================
def plot_confusion_matrix(cm, class_names, save_path):
    # plot confusion matrix with both normalized ratio and absolute count
    cm = np.array(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (Row-normalized)")
    plt.colorbar(im)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Set threshold for text color based on max normalized value
    threshold = cm_norm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value_ratio = cm_norm[i, j]
            value_count = int(cm[i, j])

            # Show both ratio and count in the cell
            text = f"{value_ratio:.2f}\n({value_count})"

            plt.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value_ratio > threshold else "black",
                fontsize=9
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



# =========================
# 10. Denormalize 
# =========================
def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


# =========================
# 11. Save misclassified examples
#     ONLY for the final best model
# =========================
@torch.no_grad()
def save_misclassified_examples(
    model,
    dataset,
    device,
    save_dir,
    class_names=None,
    max_examples: int = 16
):
    if class_names is None:
        class_names = CLASS_NAMES

    ensure_dir(save_dir)
    model.eval()

    records = []
    count = 0

    for idx in range(len(dataset)):
        image, label = dataset[idx]

        true_label = int(label.squeeze()) if hasattr(label, "squeeze") else int(label)
        image_input = image.unsqueeze(0).to(device)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            output = model(image_input)

        pred_label = int(output.argmax(dim=1).item())

        if pred_label != true_label:
            img = denormalize_image(image).permute(1, 2, 0).numpy()

            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
            plt.axis("off")

            filename = f"misclassified_{count:02d}_true_{true_label}_pred_{pred_label}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=200, bbox_inches="tight")
            plt.close()

            records.append({
                "index": idx,
                "true_label_idx": true_label,
                "true_label_name": class_names[true_label],
                "pred_label_idx": pred_label,
                "pred_label_name": class_names[pred_label],
                "filepath": filepath
            })

            count += 1
            if count >= max_examples:
                break

    pd.DataFrame(records).to_csv(
        os.path.join(save_dir, "misclassified_examples.csv"),
        index=False
    )


# =========================
# 12. Single experiment
# =========================
def run_single_experiment(
    config: dict,
    train_dataset_full,
    val_dataset,
    test_dataset,
    device,
    save_dir,
    class_names=None
):
    if class_names is None:
        class_names = CLASS_NAMES
    
    run_start_time = time.time()

    seed = config["seed"]
    fraction = config["fraction"]
    pretrained = config["pretrained"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    early_stopping_patience = config["early_stopping_patience"]

    set_seed(seed)

    init_type = "pretrained" if pretrained else "scratch"
    run_name = f"resnet50_full_{init_type}_frac{fraction}_seed{seed}"
    run_dir = os.path.join(save_dir, run_name)
    ensure_dir(run_dir)

    print(f"\n{'='*100}")
    print(f"Running: {run_name}")
    print(f"{'='*100}")

    train_subset = create_stratified_train_subset(train_dataset_full, fraction, seed)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_subset=train_subset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    model = build_resnet50_full_model(
        num_classes=len(class_names),
        pretrained=pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # IMPORTANT: Adam only, no scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\n--- [{run_name}] Epoch {epoch+1}/{num_epochs} ---")
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )

        val_loss, val_acc = evaluate_basic(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start

        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6e} | Time: {elapsed:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    run_total_time = time.time() - run_start_time   
    print(f"\nRun finished: {run_name} | Total time: {run_total_time:.2f}s ({run_total_time/60:.2f} min)")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, os.path.join(run_dir, "best_model.pth"))

    test_metrics = evaluate_full_metrics(
        model=model,
        loader=test_loader,
        device=device,
        class_names=class_names
    )

    test_acc = test_metrics["accuracy"]
    test_macro_f1 = test_metrics["macro_f1"]
    per_class_df = test_metrics["per_class_df"]
    cm = test_metrics["confusion_matrix"]

    per_class_df.to_csv(os.path.join(run_dir, "per_class_metrics.csv"), index=False)
    np.save(os.path.join(run_dir, "confusion_matrix.npy"), cm)
    plot_confusion_matrix(cm, class_names, os.path.join(run_dir, "confusion_matrix.png"))

    best_class, worst_class = get_best_worst_classes(per_class_df)

    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "fraction": fraction,
        "train_size": len(train_subset),
        "seed": seed,
        "init_type": init_type,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "best_class_name": best_class["class_name"],
        "best_class_precision": float(best_class["precision"]),
        "best_class_recall": float(best_class["recall"]),
        "best_class_f1": float(best_class["f1"]),
        "worst_class_name": worst_class["class_name"],
        "worst_class_precision": float(worst_class["precision"]),
        "worst_class_recall": float(worst_class["recall"]),
        "worst_class_f1": float(worst_class["f1"])
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


# =========================
# 13. Summary across runs
# =========================
def summarize_results(results_df: pd.DataFrame, save_dir: str):
    summary_df = (
        results_df
        .groupby(["fraction", "train_size", "init_type"], as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            std_test_macro_f1=("test_macro_f1", "std"),
            mean_best_val_acc=("best_val_acc", "mean"),
            std_best_val_acc=("best_val_acc", "std")
        )
        .sort_values(["fraction", "init_type"], ascending=[False, True])
    )

    summary_path = os.path.join(save_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary results:")
    print(summary_df)

    return summary_df


# =========================
# 14. Plot data efficiency curve
# =========================
def plot_data_efficiency(results_df: pd.DataFrame, save_dir: str):
    summary_df = (
        results_df
        .groupby(["fraction", "train_size", "init_type"], as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std")
        )
    )

    pretrained_df = summary_df[summary_df["init_type"] == "pretrained"].sort_values("train_size")
    scratch_df = summary_df[summary_df["init_type"] == "scratch"].sort_values("train_size")

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        pretrained_df["train_size"],
        pretrained_df["mean_test_accuracy"],
        yerr=pretrained_df["std_test_accuracy"],
        marker="o",
        capsize=4,
        label="Pretrained"
    )

    plt.errorbar(
        scratch_df["train_size"],
        scratch_df["mean_test_accuracy"],
        yerr=scratch_df["std_test_accuracy"],
        marker="s",
        capsize=4,
        label="From Scratch"
    )

    plt.xscale("log")
    plt.xlabel("Training Set Size (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("ResNet50 Data Efficiency: Pretrained vs From Scratch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "data_efficiency_accuracy.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 15. Save misclassified examples
#     only for overall best model
# =========================
def export_best_model_failures(
    results_df: pd.DataFrame,
    test_dataset,
    device,
    save_dir: str,
    class_names=None,
    max_examples: int = 16
):
    if class_names is None:
        class_names = CLASS_NAMES

    # Use highest test accuracy; if tie, higher val acc
    best_row = results_df.sort_values(
        by=["test_accuracy", "best_val_acc"],
        ascending=False
    ).iloc[0]

    best_run_dir = best_row["run_dir"]
    best_model_path = os.path.join(best_run_dir, "best_model.pth")

    print("\nBest overall run selected for failure analysis:")
    print(best_row[[
        "run_name", "fraction", "seed", "init_type",
        "best_val_acc", "test_accuracy", "test_macro_f1"
    ]])

    model = build_resnet50_full_model(
        num_classes=len(class_names),
        pretrained=(best_row["init_type"] == "pretrained")
    ).to(device)

    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)

    failure_dir = os.path.join(save_dir, "best_model_failure_analysis")
    ensure_dir(failure_dir)

    save_misclassified_examples(
        model=model,
        dataset=test_dataset,
        device=device,
        save_dir=failure_dir,
        class_names=class_names,
        max_examples=max_examples
    )

    with open(os.path.join(failure_dir, "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(best_row.to_dict(), f, indent=4, default=str)

    print(f"Saved best-model misclassified examples to: {failure_dir}")

