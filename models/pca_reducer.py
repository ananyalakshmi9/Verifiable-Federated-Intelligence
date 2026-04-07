import joblib
import logging
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PCAReducer:
    """
    Wrapper for Scikit-learn PCA, rigorously configured for zero-knowledge circuit 
    feasibility targets (n_components mathematically locked to circuit size limitations).
    """
    
    def __init__(self, n_components=32, random_state=42):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False
        
    def fit(self, X):
        logging.info(f"Fitting PCA. Target dimensions strictly clamped to: {self.n_components}")
        self.pca.fit(X)
        self.is_fitted = True
        
    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("PCA Reducer must be fitted before calling transform!")
        return self.pca.transform(X)
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def save(self, file_path):
        """Serialize fitted PCA object for inference parity across nodes"""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted PCA Reducer.")
        joblib.dump(self.pca, file_path)
        logging.info(f"PCA state successfully serialized and saved to {file_path}")
        
    def load(self, file_path):
        """Load serialized PCA object to maintain exact transformation matrix"""
        self.pca = joblib.load(file_path)
        self.is_fitted = True
        self.n_components = self.pca.n_components
        logging.info(f"PCA state loaded from {file_path}. Validating target dim: {self.n_components}")
        assert self.n_components == 32, "CRITICAL FAULT: Loaded PCA matrix diverges from 32-dim assumption!"
