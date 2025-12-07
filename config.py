# Auto-generated configuration file
import torch

# Hardware
DEVICE = torch.device("cpu")

# Paths
DATASET_PATH = "data/chest_xray"

# Model parameters
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ['Normal', 'Pneumonia']

# Training parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 10  # For quick testing, increase for final training
PATIENCE = 5  # Early stopping patience

# Model save paths
MODEL_SAVE_DIR = 'models'
RESULTS_DIR = 'results'
