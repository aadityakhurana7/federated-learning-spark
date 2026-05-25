import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Spark Configuration
SPARK_APP_NAME = "FederatedLearningSpark"
SPARK_MASTER = "local[*]" # Using local cluster mode for MVV

# Federated Learning Configuration
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01

# Model Configuration
# We will use Logistic Regression on a smaller dataset like MNIST or a simulated one
INPUT_SHAPE = 784 # For MNIST (28x28 flattened)
NUM_CLASSES = 10
