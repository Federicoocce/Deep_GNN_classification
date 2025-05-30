import subprocess
import os

# Base directory relative to this script
base_data_dir = "/workspace/Deep/data"

# Dataset identifiers
datasets = ["A", "B", "C", "D"]

# Script to run
main_script = os.path.join("source", "main.py")

# Run main.py for each dataset
for dataset in datasets:
    test_path = os.path.join(base_data_dir, dataset, "test.json.gz")
    train_path = os.path.join(base_data_dir, dataset, "train.json.gz")
    print(f"Running {main_script} with test path: {test_path}")
    subprocess.run(["python", main_script, "--test_path", test_path, "--train_path", train_path], check=True)
