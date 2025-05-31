import argparse
import os
from pathlib import Path

import numpy as np
import torch

from networks import MyLocalGatedGCN
from gated_gcn_training_system import GatedGCNTrainingSystem
from load_dataset import load_single_dataset
from config import get_config





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


def main():
    args = parse_args()

    # Compute base path relative to this script's location
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    logs_dir = os.path.join(base_dir, 'logs')
    submission_dir = os.path.join(base_dir, 'submission') # TODO maybe saving in zip

    # Load test data
    dataset_splits = load_single_dataset(
        train_gz_path=args.train_path,
        test_gz_path=args.test_path,
        val_split_ratio=0.1
    )

    test_data = dataset_splits['test']
    val_data = dataset_splits.get('val', None)
    train_data = dataset_splits.get('train', None)

    ds_name = get_folder_name(args.test_path or args.train_path)
    config = get_config(ds_name)

    trainer = GatedGCNTrainingSystem(
        config=config,
        model_class=MyLocalGatedGCN,
        checkpoints_path=checkpoint_dir,
        logs_path=logs_dir,
    )

    if train_data and val_data:
        trainer.train(dataset_name=ds_name,
                      train_data=train_data, val_data=val_data)
    else:
        trainer.setup_model_and_trainer()
        trainer.load(ds_name)

    trainer.generate_test_predictions(
        test_dataset=test_data,
        dataset_name=ds_name,
        output_path=submission_dir
    )

if __name__ == '__main__':
    main()
