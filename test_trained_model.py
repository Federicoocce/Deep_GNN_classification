# test_trained_model.py
import torch
# import torch.nn as nn # Not strictly needed if criterion is None for eval_epoch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import argparse
import os

# Attempt to import from sibling files
# These are the core model definition, evaluation function, and hyperparameters
try:
    from train_local_gatedgcn import MyLocalGatedGCN, eval_epoch
    from train_local_gatedgcn import (
        # Model Structure Hyperparameters (must match the trained model)
        NUM_CLASSES, GNN_LAYERS, GNN_HIDDEN_DIM, GNN_DROPOUT,
        NODE_EMBEDDING_DIM, EDGE_EMBEDDING_DIM, NODE_CATEGORY_COUNT, EDGE_FEATURE_DIM,
        # GNN "Plus" Features for StandaloneGatedGCNLayer (must match the trained model)
        USE_RESIDUAL, USE_FFN, USE_BATCHNORM,
        # DataLoader Hyperparameter
        BATCH_SIZE
    )
    from full_data_loader import get_data_splits, RWSE_MAX_K # For data loading and RWSE PE dimension
except ImportError as e:
    print(f"Critical Error: Could not import necessary components from project files: {e}")
    print("Please ensure 'train_local_gatedgcn.py' and 'full_data_loader.py' are in the same directory as this script, or correctly installed in your Python environment.")
    print("These files contain the model definition (MyLocalGatedGCN), evaluation logic (eval_epoch), and crucial hyperparameters.")
    exit(1) # Exit if essential components can't be loaded.

def run_testing(cli_args):
    """
    Loads a pre-trained model and tests it on a specified dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine PE_DIM based on whether the loaded model was trained with RWSE
    # This configuration MUST match how the model was originally trained and saved.
    model_trained_with_rwse = cli_args.model_uses_rwse
    pe_dimension_for_model = RWSE_MAX_K if model_trained_with_rwse else 0

    print(f"\n--- Model Configuration for Loading (Must match training parameters of loaded model) ---")
    print(f"  Targeting NUM_CLASSES: {NUM_CLASSES}")
    print(f"  GNN: Layers={GNN_LAYERS}, HiddenDim={GNN_HIDDEN_DIM}, Dropout={GNN_DROPOUT}")
    print(f"  Embeddings: NodeDim={NODE_EMBEDDING_DIM}, EdgeDim={EDGE_EMBEDDING_DIM}")
    print(f"  Input Features: NodeCategories={NODE_CATEGORY_COUNT}, EdgeFeatDim={EDGE_FEATURE_DIM}")
    print(f"  GNN Layer Config: Residual={USE_RESIDUAL}, FFN={USE_FFN}, BatchNorm={USE_BATCHNORM}")
    print(f"  Positional Encoding: Model Trained with RWSE = {model_trained_with_rwse}, PE_DIM = {pe_dimension_for_model}")
    print(f"-------------------------------------------------------------------------------------\n")

    # 1. Instantiate Model with the configuration it was trained with
    # MyLocalGatedGCN's __init__ takes current_use_rwse_pe and current_pe_dim.
    # Other hyperparameters like GNN_LAYERS, GNN_HIDDEN_DIM, etc., are used internally
    # by MyLocalGatedGCN by importing them from train_local_gatedgcn.py.
    model = MyLocalGatedGCN(
        current_use_rwse_pe=model_trained_with_rwse,
        current_pe_dim=pe_dimension_for_model
    ).to(device)

    # 2. Load Model Weights
    if not os.path.exists(cli_args.model_path):
        print(f"Error: Model weights file not found at '{cli_args.model_path}'. Please check the path.")
        return
    
    print(f"Attempting to load model weights from: {cli_args.model_path}")
    try:
        model.load_state_dict(torch.load(cli_args.model_path, map_location=device))
        print(f"Successfully loaded model weights.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("This often happens if the model architecture defined now (based on current hyperparameters in 'train_local_gatedgcn.py') ")
        print("does not match the architecture of the saved model. Also, ensure the '--model_uses_rwse' flag correctly reflects how the model was trained.")
        return
        
    model.eval() # Set model to evaluation mode

    # 3. Load Test Data
    selected_dataset_key = f'test_{cli_args.dataset_choice}'
    print(f"Loading data for '{selected_dataset_key}'...")
    
    all_loaded_splits = get_data_splits(force_reprocess=cli_args.force_reprocess_data) 
    
    if selected_dataset_key not in all_loaded_splits or not all_loaded_splits[selected_dataset_key]:
        print(f"Error: No test data found for dataset '{cli_args.dataset_choice}' (key: '{selected_dataset_key}'). Check 'full_data_loader.py'.")
        return
    
    test_graphs = all_loaded_splits[selected_dataset_key]
    if not test_graphs: # Check if the list of graphs is empty
        print(f"Warning: Test dataset '{selected_dataset_key}' is empty. No predictions will be generated.")
    
    # Use BATCH_SIZE imported from train_local_gatedgcn for consistency
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_graphs)} graphs for '{selected_dataset_key}'.")

    # 4. Perform Inference using eval_epoch
    print(f"\nGenerating predictions for '{selected_dataset_key}'...")
    # criterion is not used by eval_epoch when is_test_set_preds_only=True, so None is fine.
    test_predictions_array = eval_epoch(model, test_loader, criterion=None, device=device, is_test_set_preds_only=True)

    # 5. Format and Save Results
    num_test_samples = len(test_predictions_array)
    if num_test_samples > 0:
        ids = np.arange(0, num_test_samples) # Standard 1-based IDs
        predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})
        
        output_filename = cli_args.output_filename
        if not output_filename: # Generate default filename if not provided
            output_filename = f"predictions_testset_{cli_args.dataset_choice}.csv"
        
        os.makedirs(cli_args.output_dir, exist_ok=True) # Ensure output directory exists
        full_output_path = os.path.join(cli_args.output_dir, output_filename)
        
        try:
            predictions_df.to_csv(full_output_path, index=False)
            print(f"Predictions successfully saved to: {full_output_path}")
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
            
    elif len(test_graphs) > 0 : # If there was data but no predictions (e.g. model error during forward pass)
        print(f"Warning: No predictions were generated by the model, although {len(test_graphs)} test samples were loaded and processed by the DataLoader.")
        print(f"This might indicate an issue during the model's forward pass on the test data. Check for runtime errors if CUDA is used.")
    else: # No data loaded or processed by DataLoader successfully
        print(f"No test data was effectively processed for '{selected_dataset_key}', so no predictions were generated.")

    print("\nTesting complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test a pre-trained GatedGCN model on a specified dataset component (A, B, C, or D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    
    parser.add_argument('--dataset_choice', type=str, required=True, choices=['A', 'B', 'C', 'D'],
                        help="Specify which test dataset partition to use (e.g., 'A').")
    
    parser.add_argument('--model_path', type=str, default='models/best_gatedgcn_multids.pth',
                        help="Path to the saved model weights file (.pth).")
    
    parser.add_argument('--output_dir', type=str, default='predictions_output',
                        help="Directory where the output prediction CSV file will be saved.")
    
    parser.add_argument('--output_filename', type=str, default=None,
                        help="Optional: Specific name for the output CSV file. If not provided, a default name like 'predictions_testset_A.csv' is used.")
    
    parser.add_argument('--model_uses_rwse', action='store_true',
                        help="Include this flag if the model specified by --model_path was trained with RWSE (Random Walk Structural Embeddings) enabled. "
                             "The script will then instantiate the model architecture accordingly. If omitted, assumes model was trained without RWSE.")
    
    parser.add_argument('--force_reprocess_data', action='store_true',
                        help="Force the data loader to re-process the raw JSON data instead of using cached .pt files. Useful if data sources have changed or cache is suspect.")

    cli_args = parser.parse_args()
    run_testing(cli_args)