import os
import time
import json
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from . import config

def run_centralized_baseline():
    print("--- Running Centralized Baseline Training ---")
    
    # 1. Generate the exact same dataset used in FL
    X, y = make_classification(
        n_samples=10000, 
        n_features=20, 
        n_classes=2, 
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train centralized model
    start_time = time.time()
    
    model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=config.LEARNING_RATE, random_state=42)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # 3. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Centralized Model Accuracy: {acc:.4f} (Trained in {train_time:.2f}s)")
    
    # 4. Save metrics
    baseline_metrics = {
        'accuracy': acc,
        'time_taken': train_time,
        'data_transferred_bytes': X_train.nbytes + y_train.nbytes # The size of the raw dataset
    }
    
    metrics_file = os.path.join(config.LOGS_DIR, 'baseline_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(baseline_metrics, f)
        
    return baseline_metrics

if __name__ == "__main__":
    run_centralized_baseline()
