import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ZENODO_RECORD = "16736515"
API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def download_amlnet():
    logging.info(f"Querying Zenodo API for record {ZENODO_RECORD}...")
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find the main data file in the record's files
        files = data.get("files", [])
        if not files:
            logging.error("No files found in the Zenodo record.")
            raise FileNotFoundError("Empty Zenodo Record")

        dataset_file = None
        for file_info in files:
            if file_info['key'].endswith('.csv'):
                dataset_file = file_info
                break
        
        if not dataset_file:
            # Fallback if specific CSV is not found, take the first file
            dataset_file = files[0]
            
        file_name = dataset_file['key']
        download_url = dataset_file['links']['self']
        
        dest_path = os.path.join(DATA_DIR, "amlnet.csv")
        if os.path.exists(dest_path):
            logging.info(f"Dataset already exists at {dest_path}. Skipping download.")
            return
            
        logging.info(f"Downloading {file_name} from {download_url}...")
        with requests.get(download_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        logging.info(f"Download complete! Saved to {dest_path}")
        
    except Exception as e:
        logging.error(f"Error querying/downloading from Zenodo: {e}")
        # As fallback for development/demo if dataset isn't live: generate mock data
        logging.warning("Generating mock dataset for development fallback because live download failed.")
        generate_mock_dataset(os.path.join(DATA_DIR, "amlnet.csv"))

def generate_mock_dataset(dest_path):
    import pandas as pd
    import numpy as np
    import random
    
    np.random.seed(42)
    random.seed(42)
    
    # 250k total for mock data to be realistic but not overly heavy on dev tests
    num_txns = 250000 
    logging.info(f"Generating {num_txns} mock transactions...")
    
    senders = [f"A{i}" for i in range(5000)]
    receivers = [f"A{i}" for i in range(5000)]
    
    df = pd.DataFrame({
        "txn_id": range(num_txns),
        "sender": np.random.choice(senders, num_txns),
        "receiver": np.random.choice(receivers, num_txns),
        "amount": np.random.exponential(scale=1000, size=num_txns),
        "timestamp": sorted(np.random.randint(1600000000, 1630000000, num_txns)),
        "is_laundering": np.random.choice([0, 1], p=[0.98, 0.02], size=num_txns) # 2% imbalance
    })
    
    # Avoid self-loops
    df = df[df['sender'] != df['receiver']]
    df.to_csv(dest_path, index=False)
    logging.info(f"Mock dataset created at {dest_path}")

if __name__ == "__main__":
    download_amlnet()
