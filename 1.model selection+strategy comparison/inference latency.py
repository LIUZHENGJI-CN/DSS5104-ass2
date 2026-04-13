# This script measures the inference latency and throughput of the three models 
# (ResNet50, EfficientNet-B0, Swin-T) on the PathMNIST validation set. It loads the trained models,
# runs inference on a subset of the validation data, and reports the average latency per image and throughput in images per second. 
# The script uses mixed precision (AMP) for inference if a CUDA device is available to get more realistic latency measurements.

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist.dataset import PathMNIST


# define function to measure inference latency and throughput
def measure_inference_latency(model, loader, device, warmup_batches=10, measure_batches=30):
    model.eval()

    # warm-up
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                _ = model(images)

            if i + 1 >= warmup_batches:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)
            total_images += images.size(0)

            if i + 1 >= measure_batches:
                break

    latency_ms_per_image = (total_time / total_images) * 1000
    throughput_img_per_sec = total_images / total_time

    return latency_ms_per_image, throughput_img_per_sec


# define functions to build models with custom heads for 9 classes
def build_resnet50(num_classes=9):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def build_efficientnet_b0(num_classes=9):
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def build_swin_t(num_classes=9):
    model = models.swin_t(weights=None)
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


# define function to load model weights
def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


# define main function to run the latency measurements
def main():
    torch.backends.cudnn.benchmark = True

    # 1. device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 2. transform
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 3. dataset
    val_dataset = PathMNIST(split='val', transform=test_transform, download=True)

    # 4. dataloader
    batch_size = 64
    num_workers = 4

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    num_classes = 9

    # =========================
    # Paths
    # Due to GitHub file size limits, the .pth files are not included here; they will be generated in the corresponding paths when the training scripts are rerun. 
    # During the model selcetion, each model generates 3 .pth files,
    # In this section, we select full-fine tuning models because they uniformly performed best.
    resnet_path = "resnet50_full_finetuning.pth" 
    b0_path = "B0_full_finetuning.pth"
    swin_path = "swin_full_finetuning_amp.pth"

    results = {}

    # =========================
    # ResNet50
    # =========================
    print("\n" + "=" * 60)
    print("Loading ResNet50 full...")
    model_resnet = build_resnet50(num_classes=num_classes)
    model_resnet = load_model(model_resnet, resnet_path, device)
    print("ResNet50 loaded successfully.")

    latency_ms, throughput = measure_inference_latency(
        model_resnet,
        val_loader,
        device,
        warmup_batches=10,
        measure_batches=30
    )
    results["ResNet50"] = {
        "latency_ms": latency_ms,
        "throughput": throughput
    }

    print(f"ResNet50 latency (AMP): {latency_ms:.4f} ms/image")
    print(f"ResNet50 throughput (AMP): {throughput:.2f} images/second")

    # =========================
    # EfficientNet-B0
    # =========================
    print("\n" + "=" * 60)
    print("Loading EfficientNet-B0 full...")
    model_b0 = build_efficientnet_b0(num_classes=num_classes)
    model_b0 = load_model(model_b0, b0_path, device)
    print("EfficientNet-B0 loaded successfully.")

    latency_ms, throughput = measure_inference_latency(
        model_b0,
        val_loader,
        device,
        warmup_batches=10,
        measure_batches=30
    )
    results["EfficientNet-B0"] = {
        "latency_ms": latency_ms,
        "throughput": throughput
    }

    print(f"EfficientNet-B0 latency (AMP): {latency_ms:.4f} ms/image")
    print(f"EfficientNet-B0 throughput (AMP): {throughput:.2f} images/second")

    # =========================
    # Swin-T
    # =========================
    print("\n" + "=" * 60)
    print("Loading Swin-T full...")
    model_swin = build_swin_t(num_classes=num_classes)
    model_swin = load_model(model_swin, swin_path, device)
    print("Swin-T loaded successfully.")

    latency_ms, throughput = measure_inference_latency(
        model_swin,
        val_loader,
        device,
        warmup_batches=10,
        measure_batches=30
    )
    results["Swin-T"] = {
        "latency_ms": latency_ms,
        "throughput": throughput
    }

    print(f"Swin-T latency (AMP): {latency_ms:.4f} ms/image")
    print(f"Swin-T throughput (AMP): {throughput:.2f} images/second")

    # =========================
    # final comparison
    # =========================
    print("\n" + "=" * 60)
    print("Final speed comparison:")
    for name, res in results.items():
        print(f"{name}: {res['latency_ms']:.4f} ms/image | {res['throughput']:.2f} images/s")

    fastest_by_latency = min(results.items(), key=lambda x: x[1]["latency_ms"])
    fastest_by_throughput = max(results.items(), key=lambda x: x[1]["throughput"])

    print("-" * 60)
    print(f"Fastest by latency: {fastest_by_latency[0]} ({fastest_by_latency[1]['latency_ms']:.4f} ms/image)")
    print(f"Fastest by throughput: {fastest_by_throughput[0]} ({fastest_by_throughput[1]['throughput']:.2f} images/s)")


if __name__ == "__main__":
    main()