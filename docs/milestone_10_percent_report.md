# ==============================================================================
# Capstone 10% Milestone Report: Data Readiness & Preprocessing Architecture
# ==============================================================================

## 1. Requirement Analysis & Environment Setup
To build a **Decentralized AML Detection Framework** that is both secure via Federated Learning (FL) and mathematically verifiable via zk-SNARKs, the data preprocessing environment must be strictly controlled.

**Tools & Libraries Selected:**
*   **`pandas` & `numpy`**: Selected for high-performance, in-memory tabular manipulation of the raw transaction sequences. 
*   **`scikit-learn` (StandardScaler, PCA)**: Chosen for standardizing high-variance monetary inputs and providing Principal Component Analysis (PCA). PCA is absolutely critical to this architecture; our zk-SNARK compiler (`Circom`) has stringent constraints on the size of the neural network it can map into a cryptographic circuit. We enforce a rigid 32-dimension bottleneck.
*   **`imbalanced-learn` (SMOTE)**: Necessary to mathematically inflate the minority class without resorting to blind duplication.
*   **`matplotlib` & `seaborn`**: Chosen for generating comparative visual distributions to audit the integrity of the data transformation.

## 2. Data Interpretation & The Imbalance Problem
The **AMLNet dataset** accurately mirrors real-world financial topologies: legitimate transactions vastly outnumber illicit ones, resulting in an extreme class skew (often >99% legitimate to <1% illicit). 

If a neural network trains natively on this, it will suffer from "accuracy paradox"—it can achieve 99% accuracy simply by predicting *all* transactions as legitimate, entirely failing its primary objective to catch money laundering.

**Our Mitigation (Hybrid Strategy):** 
Instead of drastically undersampling (and losing critical typologies of normal financial behavior), we opted for **Local SMOTE** (Synthetic Minority Over-sampling Technique). SMOTE interprets the topological space between existing illicit nodes and synthesizes strictly contiguous neighbor-nodes, inflating the illicit presence up to 30%. This mathematically preserves the network's structural density while forcing the classifier to deeply analyze laundering patterns.

## 3. Storage Strategy
In a decentralized framework, data cannot exist in a singular location. 
*   **Partitioning Storage:** Raw datasets are aggressively partitioned into heavily isolated silos (`data/processed_partition.csv`) mirroring local banking nodes (e.g., Bank A, Bank B). 
*   **State Serialization:** Because node isolation prevents sharing centralized statistics, variables like the matrix variance (StandardScaler) and eigenvector matrices (PCA) must be securely serialized using `joblib` into `.pkl` binaries. This allows discrete nodes to apply synchronized mathematical transformations independently without accessing the broader global state.
