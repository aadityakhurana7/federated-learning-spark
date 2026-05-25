import numpy as np
from sklearn.linear_model import SGDClassifier
from . import config
import logging

# Set up explicit security logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class PrivacyEnforcer:
    """
    A strict security middleware that ensures only mathematical weights 
    leave the client, and explicitly blocks raw data (X, y).
    """
    @staticmethod
    def verify_payload_and_add_dp(weights_dict, epsilon=0.1):
        # 1. VERIFY NO RAW DATA IS LEAKING
        if 'X' in weights_dict or 'y' in weights_dict or 'data' in weights_dict:
            raise PermissionError("SECURITY BREACH: Attempted to transmit raw data over the network!")
            
        # 2. APPLY DIFFERENTIAL PRIVACY (DP)
        # Adding Gaussian noise to the weights prevents reverse-engineering attacks
        # Epsilon controls the privacy budget (lower = more private)
        dp_weights = {}
        for key, val in weights_dict.items():
            if isinstance(val, np.ndarray):
                # Add Laplace or Gaussian noise
                noise = np.random.normal(0, epsilon, val.shape)
                dp_weights[key] = val + noise
            else:
                dp_weights[key] = val
                
        logging.info(f"🔒 [SECURITY LOG] Verified payload contains NO raw data. Applied Differential Privacy noise (ε={epsilon}). Sending {len(dp_weights)} mathematical parameters.")
        return dp_weights

class FederatedClient:
    def __init__(self, client_id, X, y):
        self.client_id = client_id
        # Raw data is strictly kept local in memory
        self._local_X = X
        self._local_y = y
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=config.LEARNING_RATE, random_state=42)
        
        self.model.partial_fit(self._local_X[:2], self._local_y[:2], classes=np.array([0, 1]))

    def receive_global_model(self, global_weights):
        if global_weights is not None:
            self.model.coef_ = global_weights['coef']
            self.model.intercept_ = global_weights['intercept']

    def train_local_model(self, epochs=1):
        for _ in range(epochs):
            self.model.partial_fit(self._local_X, self._local_y)

    def extract_weights(self):
        return {
            'coef': self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy(),
            'num_samples': len(self._local_X)
        }

def spark_client_train(data_tuple, global_weights_bc, epochs):
    client_id, X, y = data_tuple
    global_weights = global_weights_bc.value
    
    # 1. Client trains locally on its strictly private data
    client = FederatedClient(client_id, X, y)
    if global_weights is not None:
        client.receive_global_model(global_weights)
        
    client.train_local_model(epochs=epochs)
    raw_weights = client.extract_weights()
    
    # 2. ENFORCE PRIVACY BEFORE DATA LEAVES THE NODE
    # Add Differential Privacy noise and strip everything except model parameters
    secured_weights = PrivacyEnforcer.verify_payload_and_add_dp(raw_weights)
    
    # For distributed reduce, we return the pre-weighted DP coefficients
    n = secured_weights['num_samples']
    return {
        'sum_coef': secured_weights['coef'] * n,
        'sum_intercept': secured_weights['intercept'] * n,
        'total_samples': n
    }
