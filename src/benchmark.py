from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate_global_model(global_weights, X_test, y_test):
    """Evaluates the aggregated global weights on the test set."""
    if global_weights is None:
        return 0.0
        
    # We create a dummy model to use sklearn's prediction logic
    dummy_model = SGDClassifier(loss='log_loss')
    # Dummy initialization
    dummy_model.partial_fit(X_test[:2], y_test[:2], classes=np.array([0, 1]))
    
    # Inject global weights
    dummy_model.coef_ = global_weights['coef']
    dummy_model.intercept_ = global_weights['intercept']
    
    # Predict and calculate accuracy
    y_pred = dummy_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return acc
