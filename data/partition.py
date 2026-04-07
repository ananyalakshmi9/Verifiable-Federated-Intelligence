import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global random seed for reproducibility
np.random.seed(42)

def partition_data():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(data_dir, "amlnet.csv")
    partitions_dir = os.path.join(data_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)
    
    if not os.path.exists(source_file):
        logging.error(f"Source data {source_file} not found. Run download_dataset.py first.")
        return
        
    logging.info(f"Loading dataset from {source_file}...")
    df = pd.read_csv(source_file)
    
    # Partitioning logic: split accounts into 4 non-overlapping groups.
    # We group by sender account to emulate actual bank customer divisions where a bank knows its clients' originations.
    unique_senders = df['sender'].unique()
    np.random.shuffle(unique_senders)
    
    num_nodes = 4
    sender_splits = np.array_split(unique_senders, num_nodes)
    
    logging.info("Partitioning into 4 non-overlapping shards based on sender accounts...")
    
    for i, seq in enumerate(sender_splits, 1):
        # Assign all transactions initiated by these senders to bank node i
        node_df = df[df['sender'].isin(seq)]
        out_path = os.path.join(partitions_dir, f"bank_node_{i}.csv")
        node_df.to_csv(out_path, index=False)
        pos_ratio = node_df['is_laundering'].mean()
        logging.info(f"Bank Node {i}: {len(node_df)} transactions, {len(seq)} unique senders. Saved to {out_path} | Distribution: {pos_ratio*100:.2f}% positive")

if __name__ == "__main__":
    partition_data()
