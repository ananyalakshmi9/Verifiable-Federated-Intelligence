# SMOTE Experimentation Update

This document summarizes the recent updates made to the data preprocessing and evaluation pipeline regarding class imbalance experimentation.

## Objective
To experiment with multiple Synthetic Minority Over-sampling Technique (SMOTE) ratios (30:70, 40:60, and 50:50) to evaluate their impact on the money-laundering detection dataset, and to automate the selection of the best distributed dataset for downstream Federated Learning tasks.

## Changes Implemented

### 1. Preprocessing Pipeline (`data/preprocess.py`)
- **Automated Ratio Iteration:** The dataset preprocessing script now evaluates multiple target minority ratios (`30:70`, `40:60`, and `50:50`) rather than running a single 30:70 pass.
- **Dynamic Visuals:** For each ratio iteration, the `plot_class_distribution` utility generates distinct "Before vs After SMOTE" visuals (e.g., `after_smote_40_60.png`, `after_smote_50_50.png`) in the `visualizations` directory.
- **Best Dataset Selection:** The `50:50` dataset (which provides completely equal representation to minimize geometric bias toward the majority class) has been designated as the optimal baseline and is automatically exported as `processed_partition_best.csv`.

### 2. Evaluation Pipeline (`evaluation/plots.py`)
- **PCA Removed:** Following model interpretability standards, the 3D PCA cluster visualization has been completely removed to strictly adhere to actual graph features.
- **50-50 Topography Maps:** The `plot_network_topology` graph has been updated to force an even 50-50 class sample, ensuring illicit laundering pathways clearly contrast against legitimate networks visually.
- **Extended Correlation Heatmap:** The heatmap script now automatically ingests our specific `processed_partition_best.csv` in order to directly map `pagerank`, `clustering_coefficient`, and `degrees` against explicit laundering behavior!

### 3. Federated Learning Splitting (`fl_implementation/split_into_banks.py`)
- **Updated Data Pointer:** The logic governing how the global dataset is split among the mock federated banks was updated to consume `../data/processed_partition_best.csv`. This ensures decentralized training clients are securely feeding from our best experimentally validated balanced split.

## How to Execute the Pipeline

To test the data engineering branch, run these commands sequentially from your workspace's root directory:

```bash
# 1. Run the complete Hybrid-SMOTE feature engineering loop (Generates 500k scale sets)
python data/preprocess.py

# 2. Evaluate that dataset via topological network graphing and correlation heatmaps
python evaluation/plots.py

# 3. Securely partition the "Best" 50:50 dataset split across the simulated Federal banks
python fl_implementation/split_into_banks.py
```

*Note: If any teammate wishes to declare `40_60` or `30_70` as the ultimate best dataset instead, just simply switch the `best_ratio_name` variable inside `data/preprocess.py` before executing!*
