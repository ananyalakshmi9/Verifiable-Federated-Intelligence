# Data Preprocessing Review & Presentation Document

This document is generated to explicitly satisfy the final year review rubric regarding Data Preprocessing. It maps your exact code modules to the academic requirements.

---

### 1. Requirement Analysis & Environment Setup
**Code Location:** `README.md`, `docker-compose.yml`, `requirements.txt`
- **Requirement Analysis:** Real-world Anti-Money Laundering (AML) architectures cannot centralize their data without violating banking regulations. Therefore, our requirement was to construct a data environment that partitions transaction graphs robustly without crossover, preserving independent silos for federated learning. Furthermore, cryptographic Zero-Knowledge (ZK) circuits enforce rigid mathematical limits, necessitating highly specialized dimensionality reduction during setup.
- **Environment Setup:** Python 3.10 virtual environments heavily dependent on tensor mathematics (`torch`) and topological mapping (`networkx`). Decentralized emulation is staged via Docker architectures restricting physical memory crossover between node containers.

### 2. Dataset, Inputs, and Tools Preparation
**Code Location:** `data/download_dataset.py`, `data/amlnet.csv`
- **Dataset:** The *AMLNet* dataset (Zenodo) acts as our synthetic base truth, representing millions of interconnected transactions mapped heavily towards a 2% illicit imbalance.
- **Inputs:** Transactions are strictly evaluated over arrays holding: `[txn_id, sender, receiver, amount, timestamp, is_laundering]`.
- **Tools:** `imbalanced-learn` (for SMOTE scaling), `scikit-learn` (for matrix normalization), and `Pandas/Numpy` combinations for structural DataFrame manipulation.

### 3. Data Pre-Processing
**Code Location:** `data/preprocess.py`, `data/hybrid_balance.py`
Data pre-processing is the largest and most complex sub-routine in the application, comprising ~20% of the architecture logic.
- **Topological Engineering:** Basic tabular transaction logs cannot capture 'money mule' laundering rings. We use `NetworkX` to actively convert ledgers into interconnected Directed Graphs to mine heuristic density values (e.g., *PageRank*, *Betweenness Centrality*).
- **Scale and Synthesis:** We execute `StandardScaler` to remove severe monetary amplitude limits, followed by a Hybrid Balancing approach employing Synthetic Minority Oversampling Technique (**SMOTE**) to pad the severe 0.2% class absence mathematically. 
- **ZK Constriction:** Due to zero-knowledge `snarkjs` compiler limitations, the entire dense matrix is actively condensed using Principal Component Analysis (**PCA**) into exactly **32 dimensions**, preventing circuit overflows.

### 4. Data Visualization
**Code Location:** `evaluation/plots.py`, `/images/`
Mathematical visualizations (using Matplotlib & Seaborn) are actively coded to render:
- **Topology Proving:** Scatter maps proving illicit transaction sequences loop heavily in high-density smurfing structures compared to diffuse legitimate transfers.
- **Spearman Matrices:** Heatmaps proving our polynomial expansion variables strongly interact with the ultimate laundering label classifier prior to PCA suppression.

### 5. Data Interpretation
**Code Location:** `data/hybrid_balance.py`
- We interpret the data's massive 98/2% skew not just as an oversampling problem, but as an error-boundary problem. By interpreting the volume algorithmically, we feed the calculated variance directly into the PyTorch backend as a variable `pos_weight`. This ensures the Loss function strictly penalizes False-Negative errors heavier than False-Positives, shifting interpretability toward regulatory risk-aversion.

### 6. Storage
**Code Location:** `data/partition.py`
- Raw `.csv` structures are aggressively split into 4 exclusive arrays and stored physically isolated within the `data/partitions/` registry. 
- Algorithmic matrices utilized during scaling (such as the fitted PCA states and StandardScaling matrices) are physically stored inside `.pkl` binary caches. This ensures testing nodes can reconstruct parameters securely via `joblib` retrieval mapping entirely blind to parallel banks.
