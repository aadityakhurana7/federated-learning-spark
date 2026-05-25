import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from . import config

def generate_and_partition_data(num_clients, num_samples=10000, n_features=20):
    """
    Generates a synthetic dataset and partitions it among clients to simulate 
    distributed datasets. Returns the global test set and a list of client datasets.
    """
    print(f"Generating synthetic dataset with {num_samples} samples and {n_features} features...")
    X, y = make_classification(
        n_samples=num_samples, 
        n_features=n_features, 
        n_classes=2, 
        random_state=42
    )
    
    # Save the generated dataset to a CSV file in the data folder for inspection
    dataset_file = os.path.join(config.DATA_DIR, "synthetic_dataset.csv")
    if not os.path.exists(dataset_file):
        print(f"Saving dataset to {dataset_file}...")
        df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
        df['label'] = y
        df.to_csv(dataset_file, index=False)
    
    # Global Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Update config input shape based on generated data
    config.INPUT_SHAPE = n_features
    config.NUM_CLASSES = 2
    
    # Partition training data among clients
    client_data = []
    chunk_size = len(X_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * chunk_size
        # For the last client, take all remaining data
        end_idx = (i + 1) * chunk_size if i < num_clients - 1 else len(X_train)
        
        X_client = X_train[start_idx:end_idx]
        y_client = y_train[start_idx:end_idx]
        
        client_data.append((X_client, y_client))
        print(f"Client {i} dataset size: {len(X_client)}")
        
    return (X_test, y_test), client_data

def get_spark_dataframe(spark, X, y):
    """Converts numpy arrays to a Spark DataFrame."""
    df_pd = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df_pd['label'] = y
    return spark.createDataFrame(df_pd)
