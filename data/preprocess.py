import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_visualizations_dir(base_dir):
    viz_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir

def plot_class_distribution(y, title, output_path):
    plt.figure(figsize=(8, 6))
    counts = pd.Series(y).value_counts()
    
    # Safely retrieve counts, defaulting to 0 if an index doesn't exist
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)
    
    labels = [f"Legitimate ({count_0})", f"Laundering ({count_1})"]
    sns.barplot(x=labels, y=[count_0, count_1], hue=labels, palette=["#2ecc71", "#e74c3c"])
    
    plt.title(title)
    plt.ylabel("Transaction Count")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    viz_dir = create_visualizations_dir(project_root)
    
    data_path = os.path.join(base_dir, "amlnet.csv")
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    logging.info(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        logging.warning("AMLNet dataset not found. Generating a hypothetical chunk for demonstration.")
        np.random.seed(42)
        X_mock = np.random.rand(10000, 40)
        y_mock = np.random.choice([0, 1], size=10000, p=[0.99, 0.01])
        df = pd.DataFrame(X_mock, columns=[f"feature_{i}" for i in range(40)])
        df['isMoneyLaundering'] = y_mock
    else:
        df = pd.read_csv(data_path)
    
    # Isolate targets and numeric features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    if 'isMoneyLaundering' in numeric_df.columns:
        X = numeric_df.drop('isMoneyLaundering', axis=1)
        y = numeric_df['isMoneyLaundering']
    else:
        raise ValueError("Target column 'isMoneyLaundering' not found in numeric columns.")

    X.fillna(0, inplace=True)

    # ---------------------------------------------------------
    # 2. Data Visualization (BEFORE)
    # ---------------------------------------------------------
    logging.info("Generating Before-SMOTE visualization...")
    plot_class_distribution(y, "Class Imbalance (Before SMOTE)", os.path.join(viz_dir, "before_smote.png"))
    
    # ---------------------------------------------------------
    # 3. Data Pre-processing: StandardScaler
    # ---------------------------------------------------------
    logging.info("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ---------------------------------------------------------
    # 4. Data Pre-processing: SMOTE
    # ---------------------------------------------------------
    logging.info("Applying SMOTE (Targeting 30% minority ratio)...")
    # For a 70/30 split, ratio = 30 / 70
    target_minority_ratio = 30 / 70
    smote = SMOTE(sampling_strategy=target_minority_ratio, k_neighbors=5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # ---------------------------------------------------------
    # 5. Data Visualization (AFTER)
    # ---------------------------------------------------------
    logging.info("Generating After-SMOTE visualization...")
    plot_class_distribution(y_resampled, "Class Balance (After SMOTE)", os.path.join(viz_dir, "after_smote.png"))
    
    # ---------------------------------------------------------
    # 6. Data Pre-processing: PCA
    # ---------------------------------------------------------
    logging.info("Applying PCA to reduce to 32 dimensions...")
    n_components = min(32, X_resampled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_resampled)
    
    processed_df = pd.DataFrame(X_pca, columns=[f"PC_{i+1}" for i in range(X_pca.shape[1])])
    processed_df['isMoneyLaundering'] = y_resampled
    
    # ---------------------------------------------------------
    # 7. Storage
    # ---------------------------------------------------------
    output_csv = os.path.join(base_dir, "processed_partition.csv")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    pca_path = os.path.join(base_dir, "pca.pkl")
    
    logging.info(f"Saving partitioned dataframe to {output_csv}...")
    processed_df.to_csv(output_csv, index=False)
    
    logging.info("Saving Scaler and PCA objects via joblib...")
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    
    logging.info("10% Implementation Pre-processing Complete!")

if __name__ == "__main__":
    main()
