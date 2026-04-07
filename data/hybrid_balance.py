import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def apply_hybrid_balancing(csv_path="amlnet_mock.csv"):
    """
    Implements a Hybrid Balancing Strategy for highly skewed AML data.
    - Local: SMOTE applied per client node to reach a 70/30 split.
    - Global: PyTorch BCEWithLogitsLoss weighted penalty for remaining skew.
    """
    print("="*50)
    print("HYBRID BALANCING STRATEGY DEMONSTRATION")
    print("="*50)
    
    # -------------------------------------------------------------
    # 1. LOAD DATASET 
    # -------------------------------------------------------------
    print("\n[1] Loading AMLNet Data...")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"    -> Warning: {csv_path} not found. Generating hypothetical AMLNet dataset with 0.2% minority class.")
        np.random.seed(42)
        num_samples = 150000
        X_mock = np.random.rand(num_samples, 32) 
        y_mock = np.random.choice([0, 1], size=num_samples, p=[0.998, 0.002]) # 0.2% laundering
        df = pd.DataFrame(X_mock, columns=[f"PC_{i}" for i in range(1, 33)])
        df['is_laundering'] = y_mock
    
    X = df.drop(columns=['is_laundering'])
    y = df['is_laundering']
    
    initial_0 = (y == 0).sum()
    initial_1 = (y == 1).sum()
    print(f"    -> Initial Distribution: Legitimate={initial_0:,}, Laundering={initial_1:,} ({initial_1/len(y)*100:.2f}%)")
    
    # -------------------------------------------------------------
    # 2. LOCAL SMOTE ( imbalanced-learn )
    # -------------------------------------------------------------
    print("\n[2] Applying Local SMOTE...")
    # The 'sampling_strategy' parameter dictates the ratio: N_minority / N_majority
    # For a 70% Major / 30% Minor split: 30/70 ≈ 0.4285
    target_minority_ratio = 30 / 70 
    print(f"    -> Target SMOTE ratio: {target_minority_ratio:.4f}")
    
    # Instantiate SMOTE. random_state ensures deterministic FL client scaling.
    smote = SMOTE(sampling_strategy=target_minority_ratio, k_neighbors=5, random_state=42)
    
    # Fit and oversample
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    resampled_0 = (y_resampled == 0).sum()
    resampled_1 = (y_resampled == 1).sum()
    print(f"    -> New SMOTE Distribution: Legitimate={resampled_0:,}, Laundering={resampled_1:,} ({resampled_1/len(y_resampled)*100:.2f}%)")
    
    # Plot generation for verification
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=['Legitimate', 'Laundering'], y=[initial_0, initial_1], ax=axes[0], palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title(f"Original Skew ({initial_1/len(y)*100:.2f}%)")
    axes[0].set_ylabel("Transactions")
    
    sns.barplot(x=['Legitimate', 'Laundering'], y=[resampled_0, resampled_1], ax=axes[1], palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title(f"After SMOTE Target ({resampled_1/len(y_resampled)*100:.2f}%)")
    
    output_img = "/Users/ananyalakshmi/.gemini/antigravity/brain/49290664-1efd-4fc0-85d6-1d5598d16fde/smote_verification.png"
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"    -> Verification Chart saved to: {output_img}")
    
    # -------------------------------------------------------------
    # 3. GLOBAL CLASS WEIGHTING ( PyTorch )
    # -------------------------------------------------------------
    print("\n[3] Calculating PyTorch Cost-Sensitive Weights...")
    # Even safely padded to 30%, it is not 50/50. 
    # The PyTorch pos_weight formula dictates: pos_weight = majority_count / minority_count
    pos_weight_val = resampled_0 / resampled_1
    print(f"    -> Computed pos_weight: {pos_weight_val:.4f}")
    
    # Convert to Tensor dynamically depending on deployment device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    
    # IMPORTANT DEVIATION FOR IMPLEMENTATION:
    # BCEWithLogitsLoss is far more numerically stable than calling nn.Sigmoid() + nn.BCELoss()
    # Ensure your MLP's final layer is just nn.Linear(32, 1) without the nn.Sigmoid() activation!
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    print(f"    -> Success: PyTorch BCEWithLogitsLoss initialized with dynamic weight penalties.")
    print("="*50)
    
    return X_resampled, y_resampled, criterion

if __name__ == "__main__":
    apply_hybrid_balancing()
