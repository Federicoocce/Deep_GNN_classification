try:
    from data_loader import RWSE_MAX_K
except ImportError:
    print("Warning: Could not import RWSE_MAX_K from data_loader. Using a placeholder value (e.g., 16).")
    RWSE_MAX_K = 16  # Placeholder if import fails


class ModelConfig:


    """Configuration class for model hyperparameters"""
    # Model Architecture
    NUM_CLASSES = 6
    GNN_LAYERS = 3
    GNN_HIDDEN_DIM = 256
    GNN_DROPOUT = 0.4
    NODE_EMBEDDING_DIM = 128
    EDGE_EMBEDDING_DIM = 128

    # Data dimensions
    NODE_CATEGORY_COUNT = 1
    EDGE_FEATURE_DIM = 7

    # Positional Encoding
    USE_RWSE_PE = False
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0  # This will be dynamically updated in get_config_from_args

    # GNN Features
    USE_RESIDUAL = True
    USE_FFN = False
    USE_BATCHNORM = True

    # Training
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1.0e-5
    EPOCHS = 300
    BATCH_SIZE = 32
    NUM_WARMUP_EPOCHS = 10
    LABEL_SMOOTHING = 0.0
    EDGE_DROPPING = 0.1
    LOSS_FUNCTION = 'ce'  # Default loss function
    GCE_Q = 0.5  # Default GCE q value
    SCE_ALPHA = 0.1  # Default SCE alpha value
    SCE_BETA = 1.0  # Default SCE beta value

    # Early Learning Regularization
    USE_ELR = False
    ELR_BETA = 0.7  # Default beta for ELR


    # Early Stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 50
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Co-Teaching
    CO_TEACHING = False
    MAX_CO_TEACHING_FORGET_RATE = 0.40
    NUM_GRADUAL = 5
    FORGET_RATE_MAX_EPOCH = 30
    EXPONENT = 0.25

    # Cleanlab (new defaults)
    CLEANLAB_EPOCHS = 10
    CLEANLAB_LR = 1e-4
    CLEANLAB_BS_MULTIPLIER = 1  # Multiplier for the main batch_size for Cleanlab's temp training

def get_config(ds_name):
    """
    Returns ModelConfig object with dataset-specific parameters.

    Args:
        ds_name (str): Dataset name ('A', 'B', 'C', or 'D')

    Returns:
        ModelConfig: Configuration object with dataset-specific parameters
    """
    config = ModelConfig()

    # Dataset-specific configurations
    if ds_name == 'A':
        config.GNN_HIDDEN_DIM = 256
        config.LEARNING_RATE = 0.0005
        config.EPOCHS = 300
        config.BATCH_SIZE = 32
        config.GNN_LAYERS = 3
        config.NODE_EMBEDDING_DIM = 128
        config.EDGE_EMBEDDING_DIM = 128
        config.GNN_DROPOUT = 0.5
        config.USE_RESIDUAL = True
        config.USE_FFN = False
        config.USE_BATCHNORM = True
        config.USE_RWSE_PE = False
        config.PE_DIM = 0
        config.EDGE_DROPPING = 0.2
        # Loss function: gce q=0.5
        config.LOSS_FUNCTION = 'gce'
        config.GCE_Q = 0.5

    elif ds_name == 'B':
        config.GNN_HIDDEN_DIM = 128
        config.LEARNING_RATE = 0.0005
        config.EPOCHS = 300
        config.BATCH_SIZE = 32
        config.GNN_LAYERS = 3
        config.NODE_EMBEDDING_DIM = 128
        config.EDGE_EMBEDDING_DIM = 128
        config.GNN_DROPOUT = 0.5
        config.USE_RESIDUAL = True
        config.USE_FFN = False
        config.USE_BATCHNORM = True
        config.USE_RWSE_PE = False
        config.PE_DIM = 0
        config.EDGE_DROPPING = 0.2
        # Loss function: gce q=0.9
        config.LOSS_FUNCTION = 'gce'
        config.GCE_Q = 0.9

    elif ds_name == 'C':
        config.GNN_HIDDEN_DIM = 512
        config.LEARNING_RATE = 0.0005
        config.EPOCHS = 300
        config.BATCH_SIZE = 32
        config.GNN_LAYERS = 3
        config.NODE_EMBEDDING_DIM = 256
        config.EDGE_EMBEDDING_DIM = 256
        config.GNN_DROPOUT = 0.5
        config.USE_RESIDUAL = True
        config.USE_FFN = False
        config.USE_BATCHNORM = True
        config.USE_RWSE_PE = False
        config.PE_DIM = 0
        config.EDGE_DROPPING = 0.2
        # Loss function: ce (LS: 0.1)
        config.LOSS_FUNCTION = 'ce'
        config.LABEL_SMOOTHING = 0.1

    elif ds_name == 'D':
        config.GNN_HIDDEN_DIM = 256
        config.LEARNING_RATE = 0.0005
        config.EPOCHS = 300
        config.BATCH_SIZE = 32
        config.GNN_LAYERS = 3
        config.NODE_EMBEDDING_DIM = 256
        config.EDGE_EMBEDDING_DIM = 256
        config.GNN_DROPOUT = 0.5
        config.USE_RESIDUAL = True
        config.USE_FFN = False
        config.USE_BATCHNORM = True
        config.USE_RWSE_PE = False
        config.PE_DIM = 0
        config.EDGE_DROPPING = 0.25  # Higher edge drop for dataset D
        # Loss function: sce alfa 0.1 beta 1
        config.LOSS_FUNCTION = 'sce'
        config.SCE_ALPHA = 0.1
        config.SCE_BETA = 1.0

    else:
        raise ValueError(f"Unknown dataset name: {ds_name}. Expected one of ['A', 'B', 'C', 'D']")

    return config
