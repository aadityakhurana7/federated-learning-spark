import numpy as np

class FederatedServer:
    def __init__(self):
        self.global_weights = None
        self.history = []

    def aggregate_weights(self, reduced_weights):
        """
        Completes the FedAvg algorithm after Spark has performed distributed reduction.
        """
        total_samples = reduced_weights['total_samples']
        
        # Divide the summed coefficients by the total number of samples
        agg_coef = reduced_weights['sum_coef'] / total_samples
        agg_intercept = reduced_weights['sum_intercept'] / total_samples
            
        self.global_weights = {
            'coef': agg_coef,
            'intercept': agg_intercept
        }
        
        return self.global_weights

    def get_global_weights(self):
        return self.global_weights
