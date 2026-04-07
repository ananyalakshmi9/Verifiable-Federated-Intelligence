import networkx as nx
import pandas as pd
import numpy as np
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_features(df):
    """
    Extracts topological features using networkx.
    Input df must have cols: [txn_id, sender, receiver, amount, timestamp, is_laundering]
    """
    logging.info(f"Starting graph feature extraction for {len(df)} transactions...")
    
    # Build Directed Graph
    G = nx.from_pandas_edgelist(
        df, 
        source='sender', 
        target='receiver', 
        edge_attr=['amount', 'txn_id', 'timestamp'], 
        create_using=nx.DiGraph()
    )
    
    logging.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Calculate Node-level features
    logging.info("Calculating PageRank (damping=0.85)...")
    pagerank_dict = nx.pagerank(G, alpha=0.85, weight='amount')
    
    logging.info("Calculating In/Out Degrees...")
    in_degree_dict = dict(G.in_degree(weight=None))
    out_degree_dict = dict(G.out_degree(weight=None))
    
    logging.info("Calculating Clustering Coefficient...")
    # Unweighted directed clustering coefficient computation
    clustering_dict = nx.clustering(nx.DiGraph(G))
    
    logging.info("Calculating Betweenness Centrality (normalized approx)...")
    # k specifies subset of nodes for approx efficiency, normalized by default
    k = min(100, G.number_of_nodes())
    betweenness_dict = nx.betweenness_centrality(G, k=k, weight='amount', seed=42)
    
    # Assemble Node Features DataFrame
    node_features = pd.DataFrame({
        'account': list(G.nodes()),
        'pagerank': [pagerank_dict.get(n, 0) for n in G.nodes()],
        'in_degree': [in_degree_dict.get(n, 0) for n in G.nodes()],
        'out_degree': [out_degree_dict.get(n, 0) for n in G.nodes()],
        'clustering_coefficient': [clustering_dict.get(n, 0) for n in G.nodes()],
        'betweenness_centrality': [betweenness_dict.get(n, 0) for n in G.nodes()],
    })
    
    logging.info("Mapping node features to edges internally...")
    
    # Ensure sequential timestamping for temporal_delta
    df = df.sort_values(by=['sender', 'timestamp'])
    
    # Calculate temporal_delta: time since last txn from SAME sender
    df['temporal_delta'] = df.groupby('sender')['timestamp'].diff().fillna(0)
    
    # Log-Transform the amount to reduce large transaction variance skewness
    df['amount_log'] = df['amount'].apply(lambda x: math.log1p(x) if x > 0 else 0)
    
    # Merge Node Features for Senders
    df = df.merge(
        node_features.add_suffix('_sender'), 
        left_on='sender', 
        right_on='account_sender', 
        how='left'
    ).drop(columns=['account_sender'])
    
    # Merge Node Features for Receivers
    df = df.merge(
        node_features.add_suffix('_receiver'), 
        left_on='receiver', 
        right_on='account_receiver', 
        how='left'
    ).drop(columns=['account_receiver'])
    
    # Fill any NaNs resulting from merges just in case
    df = df.fillna(0)
    
    logging.info("Feature extraction successfully completed.")
    return df
