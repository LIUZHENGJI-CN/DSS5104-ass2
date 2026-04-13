# This is the main script to run all the ResNet50 data efficiency experiments.

import os
import pandas as pd
import torch

from resnet_data_efficiency_utils import (
    CLASS_NAMES,
    ensure_dir,
    load_datasets,
    run_single_experiment,
    summarize_results,
    plot_data_efficiency,
    export_best_model_failures
)


# =========================
# 1. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. Global save dir
# =========================
SAVE_DIR = "experiment result"
ensure_dir(SAVE_DIR)


# =========================
# 3. Base config
#    You can change these safely
# =========================
BASE_CONFIG = {
    "batch_size": 64,
    "num_workers": 4,
    "num_epochs": 8,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "early_stopping_patience": 3
}


# =========================
# 4. Experiment combinations
#    Modify here for controllable runs
# =========================
fractions = [1.0, 0.5, 0.25, 0.1, 0.05]
seeds = [42, 52, 62]
pretrained_options = [True, False]   # True = pretrained, False = scratch


# =========================
# 5. Build config list
# =========================
def build_experiment_configs():
    configs = []

    for fraction in fractions:
        for seed in seeds:
            for pretrained in pretrained_options:
                cfg = BASE_CONFIG.copy()
                cfg.update({
                    "fraction": fraction,
                    "seed": seed,
                    "pretrained": pretrained
                })
                configs.append(cfg)

    return configs


# =========================
# 6. Main
# =========================
def main():
    train_dataset_full, val_dataset, test_dataset = load_datasets(use_train_aug=True)

    print(f"Device: {device}")
    print(f"Full train size: {len(train_dataset_full)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    configs = build_experiment_configs()
    all_results = []

    for i, config in enumerate(configs, start=1):
        print(f"\n##### Running config {i}/{len(configs)} #####")
        print(config)

        result = run_single_experiment(
            config=config,
            train_dataset_full=train_dataset_full,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            save_dir=SAVE_DIR,
            class_names=CLASS_NAMES
        )
        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            os.path.join(SAVE_DIR, "all_results_raw.csv"),
            index=False
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(SAVE_DIR, "all_results_raw.csv"), index=False)

    summarize_results(results_df, SAVE_DIR)
    plot_data_efficiency(results_df, SAVE_DIR)

    # Only export misclassified examples for the best overall model
    export_best_model_failures(
        results_df=results_df,
        test_dataset=test_dataset,
        device=device,
        save_dir=SAVE_DIR,
        class_names=CLASS_NAMES,
        max_examples=16
    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()