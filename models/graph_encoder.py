import networkx as nx
import numpy as np
import logging
try:
    from node2vec import Node2Vec
except ImportError:
    Node2Vec = None # Handle graceful degradation if environment isn't fully installed yet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GraphEncoder:
    """
    Sub-module focused purely on deep topological embeddings (Node2Vec).
    Complements the core metrics (PageRank, Centrality) extracted in data/graph_features.py
    """
    def __init__(self, dimensions=16, walk_length=10, num_walks=5, workers=1):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers # Hard-set to 1 to strictly abide by reproducibility requirements
        self.n2v_model = None
        
    def fit_transform(self, G: nx.DiGraph):
        if Node2Vec is None:
            logging.error("Node2Vec library not found. Falling back to zeros matrix.")
            return {str(node): np.zeros(self.dimensions) for node in G.nodes()}
            
        logging.info("Encoding Deep Graph Topology with Node2Vec...")
        
        # Strict seed applied for reproducible walk generations
        node2vec = Node2Vec(G, dimensions=self.dimensions, walk_length=self.walk_length, 
                            num_walks=self.num_walks, workers=self.workers, seed=42)
                            
        # Train word2vec core
        self.n2v_model = node2vec.fit(window=5, min_count=1, batch_words=4)
        logging.info("Node2Vec encoding complete.")
        
        embeddings = {str(node): self.n2v_model.wv[str(node)] for node in G.nodes()}
        return embeddings
        
    def save_model(self, path):
        if self.n2v_model:
            self.n2v_model.save(path)
            logging.info(f"Node2Vec saved to {path}")
            
    def load_model(self, path):
        from gensim.models import Word2Vec
        self.n2v_model = Word2Vec.load(path)
        logging.info(f"Node2Vec loaded from {path}")
