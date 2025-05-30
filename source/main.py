import argparse
import os

import numpy as np
import torch

from networks import MyLocalGatedGCN
from source.gated_gcn_training_system import GatedGCNTrainingSystem
from source.load_dataset import load_single_dataset

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

    # Early Learning Regularization
    USE_ELR = False
    ELR_BETA = 0.7  # Default beta for ELR

    # Early Stopping
    EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 30
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


def parse_args():
    parser = argparse.ArgumentParser(description="Hackathon Graph Noisy Labels")
    parser.add_argument('--test_path', type=str, required=True, help='Path to test.json.gz')
    parser.add_argument('--train_path', type=str, default=None, help='Optional path to train.json.gz')
    return parser.parse_args()


def get_folder_name(path):
    # Extracts the folder name (e.g., A, B, C, D) from the path
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part in ['A', 'B', 'C', 'D']:
            return part
    raise ValueError("Folder name (A, B, C, D) not found in the provided path.")


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
        config.PREDICT_ON_TEST = False

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
        config.PREDICT_ON_TEST = False

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
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    # Load test data
    dataset_splits = load_single_dataset(train_gz_path=args.train_path, test_gz_path=args.test_path,
                                         val_split_ratio=0.2)  # TODO val split

    test_data = dataset_splits['test']
    val_data = dataset_splits.get('val', None)
    train_data = dataset_splits.get('train', None)

    ds_name = get_folder_name(args.test_path or args.train_path)
    config = get_config(ds_name)

    checkpoint_dir = os.path.join('../checkpoints')
    logs_dir = os.path.join('../logs')
    submission_dir = os.path.join('../submissions')
    trainer = GatedGCNTrainingSystem(
        config=config,
        model_class=MyLocalGatedGCN,
        checkpoints_path=checkpoint_dir,
        logs_path=logs_dir,
    )
    if train_data and val_data:
        trainer.train(dataset_name=ds_name,
                      train_data=train_data, val_data=val_data,
                      loss_fn_name=config.LOSS_FUNCTION)  # TODO loss
    else:
        trainer.setup_model_and_trainer()
        trainer.load(ds_name)

    trainer.generate_test_predictions(
        test_dataset=test_data,
        dataset_name=ds_name,
        output_path=os.path.join(submission_dir)
    )


if __name__ == '__main__':
    main()
