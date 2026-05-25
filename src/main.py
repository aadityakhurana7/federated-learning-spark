import os
import time
import json
import pickle
from src import config
from src.spark_manager import init_spark
from src.data_loader import generate_and_partition_data
from src.client import spark_client_train
from src.server import FederatedServer
from src.benchmark import evaluate_global_model
from src.centralized_baseline import run_centralized_baseline

def run_federated_learning():
    # Run the centralized baseline first for comparison
    run_centralized_baseline()
    
    print("\nInitializing Spark...")
    spark = init_spark()
    sc = spark.sparkContext
    
    # 1. FAULT TOLERANCE: Set Spark checkpoint directory to truncate lineage
    checkpoint_dir = os.path.join(config.LOGS_DIR, "spark_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sc.setCheckpointDir(checkpoint_dir)
    
    print("Generating and partitioning data...")
    (X_test, y_test), client_data = generate_and_partition_data(num_clients=config.NUM_CLIENTS)
    
    # Bundle data into (client_id, X, y)
    base_data = [(i, data[0], data[1]) for i, data in enumerate(client_data)]
    
    # 2. SPARK CACHING: Create RDD once and cache it in memory across worker nodes
    # This avoids network transfer of raw data on every round
    client_data_rdd = sc.parallelize(base_data, numSlices=config.NUM_CLIENTS).cache()
    
    server = FederatedServer()
    metrics = {'rounds': [], 'accuracy': [], 'time_taken': []}
    
    print(f"Starting Federated Learning for {config.NUM_ROUNDS} rounds...")
    
    for round_num in range(1, config.NUM_ROUNDS + 1):
        start_time = time.time()
        print(f"\n--- Round {round_num} ---")
        
        global_weights = server.get_global_weights()
        
        # 3. SPARK BROADCASTING: Efficiently send the global model to all workers
        global_weights_bc = sc.broadcast(global_weights)
        epochs = config.LOCAL_EPOCHS
        
        # Train locally on workers
        trained_rdd = client_data_rdd.map(lambda data_tuple: spark_client_train(data_tuple, global_weights_bc, epochs))
        
        # 4. DISTRIBUTED AGGREGATION: Reduce happens in parallel on workers
        reduced_weights = trained_rdd.reduce(
            lambda a, b: {
                'sum_coef': a['sum_coef'] + b['sum_coef'],
                'sum_intercept': a['sum_intercept'] + b['sum_intercept'],
                'total_samples': a['total_samples'] + b['total_samples']
            }
        )
        
        # Driver finalizes the aggregation
        new_global_weights = server.aggregate_weights(reduced_weights)
        
        # Fault Tolerance: Save Global Model Checkpoint
        if round_num % 5 == 0:
            with open(os.path.join(config.LOGS_DIR, f"model_cp_round_{round_num}.pkl"), "wb") as f:
                pickle.dump(new_global_weights, f)
        
        # Evaluate global model
        acc = evaluate_global_model(new_global_weights, X_test, y_test)
        
        round_time = time.time() - start_time
        print(f"Round {round_num} completed in {round_time:.2f}s. Global Accuracy: {acc:.4f}")
        
        # Store metrics (including the live model brain for the frontend)
        metrics['rounds'].append(round_num)
        metrics['accuracy'].append(acc)
        metrics['time_taken'].append(round_time)
        metrics['latest_weights'] = new_global_weights['coef'].flatten().tolist()
        
        with open(os.path.join(config.LOGS_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
            
    print("\nFederated Learning Training Complete.")
    
    # Keep the Spark context alive so you can inspect the Spark UI
    print("\n" + "="*60)
    print("🔍 SPARK UI IS ACTIVE: Open http://localhost:4040 in your browser")
    print("Check the 'Storage' tab for cached data and 'Stages' for DAG visualization.")
    print("="*60)
    input("\nPress Enter to stop the Spark cluster and exit...")
    
    spark.stop()

if __name__ == "__main__":
    run_federated_learning()
