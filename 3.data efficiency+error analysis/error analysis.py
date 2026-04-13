import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from medmnist import PathMNIST


# =========================================================
# 1. Paths
# Due to GitHub file size limits, the .pth files are not included here; they will be generated in the corresponding paths when the training scripts are rerun. 
# After data efficiency experiments, a folder named 'experiment result' will be generated, please check the following path(under this folder) to load the best model for error analysis:

MODEL_PATH = "experiment result/resnet50_full_pretrained_frac1.0_seed42/best_model.pth"
OUTPUT_DIR = "targeted_error_analysis"


# =========================================================
# 2. Class names
# =========================================================
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


# =========================================================
# 3. Target strategy: total 20 errors
# =========================================================
TARGET_COUNTS = {
    ("cancer_associated_stroma", "smooth_muscle"): 5,
    ("cancer_associated_stroma", "colorectal_adenocarcinoma_epithelium"): 3,
    ("cancer_associated_stroma", "debris"): 3,
    ("mucus", "adipose"): 2,
    ("normal_colon_mucosa", "colorectal_adenocarcinoma_epithelium"): 2,
    ("adipose", "smooth_muscle"): 2,
    ("smooth_muscle", "cancer_associated_stroma"): 1,
    ("debris", "smooth_muscle"): 1,
    ("normal_colon_mucosa", "lymphocytes"): 1,
}


# =========================================================
# 4. Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 5. Utilities
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_resnet50_full_model(num_classes=9):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


# =========================================================
# 6. Main
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    image_dir = os.path.join(OUTPUT_DIR, "selected_images")
    ensure_dir(image_dir)

    # test-set logic aligned with your original code
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = PathMNIST(
        split="test",
        transform=test_transform,
        download=True
    )

    model = build_resnet50_full_model(num_classes=len(CLASS_NAMES)).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Device: {device}")
    print(f"Test size: {len(test_dataset)}")

    # -----------------------------------------------------
    # Step 1: run inference on the whole test set
    # -----------------------------------------------------
    records = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]

            true_label = int(label.squeeze()) if hasattr(label, "squeeze") else int(label)

            x = image.unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            pred_label = int(torch.argmax(probs, dim=1).item())
            pred_confidence = float(probs[0, pred_label].item())
            true_class_confidence = float(probs[0, true_label].item())

            records.append({
                "index": idx,
                "true_label_idx": true_label,
                "true_label_name": CLASS_NAMES[true_label],
                "pred_label_idx": pred_label,
                "pred_label_name": CLASS_NAMES[pred_label],
                "pred_confidence": pred_confidence,
                "true_class_confidence": true_class_confidence,
                "is_misclassified": pred_label != true_label
            })

            if (idx + 1) % 1000 == 0 or (idx + 1) == len(test_dataset):
                print(f"Inference progress: {idx + 1}/{len(test_dataset)}")

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "full_test_predictions.csv"), index=False)

    test_acc = (pred_df["true_label_idx"] == pred_df["pred_label_idx"]).mean()
    print(f"\nRecomputed test accuracy: {test_acc:.4f}")

    # -----------------------------------------------------
    # Step 2: targeted selection of 20 misclassified images
    #         pick highest-confidence wrong predictions first
    # -----------------------------------------------------
    mis_df = pred_df[pred_df["is_misclassified"]].copy()

    selected_parts = []
    shortage_info = []

    for (true_name, pred_name), need_n in TARGET_COUNTS.items():
        subset = mis_df[
            (mis_df["true_label_name"] == true_name) &
            (mis_df["pred_label_name"] == pred_name)
        ].copy()

        subset = subset.sort_values(
            by=["pred_confidence", "true_class_confidence"],
            ascending=[False, True]
        )

        chosen = subset.head(need_n).copy()
        chosen["selection_group"] = f"{true_name} -> {pred_name}"
        selected_parts.append(chosen)

        if len(chosen) < need_n:
            shortage_info.append({
                "pair": f"{true_name} -> {pred_name}",
                "needed": need_n,
                "selected": len(chosen)
            })

    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df.to_csv(
        os.path.join(OUTPUT_DIR, "selected_20_errors_raw.csv"),
        index=False
    )

    print(f"Selected {len(selected_df)} targeted misclassified examples.")

    if shortage_info:
        print("\nWarning: some target pairs had fewer samples than requested:")
        for item in shortage_info:
            print(item)

    # -----------------------------------------------------
    # Step 3: save the 20 selected images
    # -----------------------------------------------------
    save_records = []

    for i, row in selected_df.iterrows():
        idx = int(row["index"])
        image, label = test_dataset[idx]

        img = denormalize_image(image).permute(1, 2, 0).numpy()

        true_name = row["true_label_name"]
        pred_name = row["pred_label_name"]
        true_idx = int(row["true_label_idx"])
        pred_idx = int(row["pred_label_idx"])
        pred_conf = float(row["pred_confidence"])
        group_name = row["selection_group"]

        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(img)
        plt.title(
            f"{group_name}\n"
            f"True: {true_idx} ({true_name})\n"
            f"Pred: {pred_idx} ({pred_name}) | Conf: {pred_conf:.3f}",
            fontsize=8
        )
        plt.axis("off")

        filename = f"{i:02d}_idx{idx}_true{true_idx}_pred{pred_idx}.png"
        filepath = os.path.join(image_dir, filename)

        plt.savefig(filepath, dpi=220, bbox_inches="tight")
        plt.close()

        record = row.to_dict()
        record["saved_path"] = filepath
        save_records.append(record)

    final_df = pd.DataFrame(save_records)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "selected_20_errors.csv"), index=False)

    print(f"\nDone. Images saved to: {image_dir}")
    print(f"CSV saved to: {os.path.join(OUTPUT_DIR, 'selected_20_errors.csv')}")


if __name__ == "__main__":
    main()