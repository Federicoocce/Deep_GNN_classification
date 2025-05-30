# gated_gcn_training_system.py
import os
import random
import time
import copy
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

# Assuming these are in the same directory or proper PYTHON PATH is set
from training_utils import EarlyStopping  # drop_edges is used by trainers
from losses import LossManager, ELRLoss  # ELRLoss for type checking and direct instantiation
from trainers import SingleModelTrainer, CoTeachingTrainer
from cleanlab_processor import CleanLabProcessor


def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility across common libraries and configure deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


class GatedGCNTrainingSystem:
    def __init__(self, config, model_class, checkpoints_path, logs_path):
        """
        Initialize the training system.

        Args:
            config: Configuration object with training parameters
            model_class: Model class to instantiate (e.g., MyLocalGatedGCN)
            checkpoints_path: Path to directory where checkpoints will be saved
            logs_path: Path to directory where training logs will be saved
        """
        set_seed()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_class = model_class
        self.checkpoints_path = checkpoints_path
        self.logs_path = logs_path

        # Create directories if they don't exist
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

        self.model = None  # For single model
        self.model1 = None  # For co-teaching
        self.model2 = None  # For co-teaching

        self.teacher_model = None  # For ELR single model
        self.trainer = None

        # For logging
        self.current_dataset_name = None
        self.training_logs = []

    def setup_model_and_trainer(self):
        """Setup models and trainer based on configuration."""
        if self.config.CO_TEACHING:
            self.model1 = self.model_class(self.config).to(self.device)
            self.model2 = self.model_class(self.config).to(self.device)
            self.trainer = CoTeachingTrainer(self.model1, self.model2, self.config, self.device)
            num_params = sum(p.numel() for p in self.model1.parameters() if p.requires_grad)
            print(f"Number of trainable params per model (Co-Teaching): {num_params:,}")
        else:
            self.model = self.model_class(self.config).to(self.device)
            if self.config.USE_ELR:
                self.teacher_model = copy.deepcopy(self.model)
            self.trainer = SingleModelTrainer(self.model, self.config, self.device)
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters (Single Model): {num_params:,}")

    def _validate_data_dimensions(self, train_data):
        """Validate that data dimensions match configuration."""
        if (train_data and hasattr(train_data[0], 'edge_attr') and
                train_data[0].edge_attr is not None and train_data[0].edge_attr.numel() > 0):
            actual_edge_dim = train_data[0].edge_attr.shape[1]
            if actual_edge_dim != self.config.EDGE_FEATURE_DIM:
                print(
                    f"WARNING: Edge dim mismatch. Data: {actual_edge_dim}, Model expects: {self.config.EDGE_FEATURE_DIM}.")

    def create_data_loaders(self, train_data, val_data=None):
        """Create data loaders from provided data."""
        train_loader = None
        if train_data and len(train_data) > 0:
            train_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE,
                                      shuffle=True, num_workers=getattr(self.config, 'NUM_WORKERS', 0))

        val_loader = None
        if val_data and len(val_data) > 0:
            val_loader = DataLoader(val_data, batch_size=self.config.BATCH_SIZE,
                                    shuffle=False, num_workers=getattr(self.config, 'NUM_WORKERS', 0))
        return train_loader, val_loader

    def create_optimizers_and_schedulers(self):
        """Create optimizers and learning rate schedulers."""
        params1, params2 = None, None
        if self.config.CO_TEACHING:
            params1 = self.model1.parameters()
            params2 = self.model2.parameters()
        else:
            params1 = self.model.parameters()

        optimizer1 = torch.optim.AdamW(params1, lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        optimizer2 = None
        if self.config.CO_TEACHING:
            optimizer2 = torch.optim.AdamW(params2, lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)

        def lr_lambda_fn(current_epoch_internal):
            if current_epoch_internal < self.config.NUM_WARMUP_EPOCHS:
                return float(current_epoch_internal + 1) / float(self.config.NUM_WARMUP_EPOCHS + 1)
            progress = float(current_epoch_internal - self.config.NUM_WARMUP_EPOCHS) / \
                       float(max(1, self.config.EPOCHS - self.config.NUM_WARMUP_EPOCHS))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler1, scheduler2 = None, None
        if self.config.EPOCHS > self.config.NUM_WARMUP_EPOCHS:
            scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda_fn)
            if self.config.CO_TEACHING and optimizer2:
                scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda_fn)

        if self.config.CO_TEACHING:
            return (optimizer1, optimizer2), (scheduler1, scheduler2)
        return optimizer1, scheduler1

    def _save_checkpoint(self, epoch, dataset_name):
        """Save model checkpoint for given epoch."""
        if self.config.CO_TEACHING:
            model1_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{epoch}_model1.pth')
            model2_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{epoch}_model2.pth')
            if self.model1:
                torch.save(self.model1.state_dict(), model1_path)
            if self.model2:
                torch.save(self.model2.state_dict(), model2_path)
        else:
            model_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{epoch}.pth')
            if self.model:
                torch.save(self.model.state_dict(), model_path)

    def _save_training_log(self, dataset_name):
        """Save training logs to JSON file."""
        log_file = os.path.join(self.logs_path, f'training_log_{dataset_name}.json')
        with open(log_file, 'w') as f:
            json.dump(self.training_logs, f, indent=2)

    def _log_epoch(self, epoch, metrics, elapsed_time):
        """Log epoch information."""
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            'elapsed_time': elapsed_time,
            **metrics
        }
        self.training_logs.append(log_entry)

    def _print_config(self, dataset_name, loss_name_str, criterion_instance, use_elr_flag):
        """Print training configuration."""
        print(f"\n--- Configuration ---")
        print(f"Dataset: {dataset_name}, Device: {self.device}")
        print(f"Training Mode: {'Co-Teaching' if self.config.CO_TEACHING else 'Single Model'}")
        if self.config.CO_TEACHING:
            print(
                f"  CoT Forget: NumGradual={self.config.NUM_GRADUAL}, Exp={self.config.EXPONENT}, MaxRate={self.config.MAX_CO_TEACHING_FORGET_RATE}, MaxEpochRate={self.config.FORGET_RATE_MAX_EPOCH}")
        print(
            f"Epochs: {self.config.EPOCHS}, LR: {self.config.LEARNING_RATE}, Batch: {self.config.BATCH_SIZE}, WD: {self.config.WEIGHT_DECAY}")
        print(f"Warmup Epochs: {self.config.NUM_WARMUP_EPOCHS}, Edge Dropping: {self.config.EDGE_DROPPING}")
        print(
            f"Model: {self.model_class.__name__}, Layers={self.config.GNN_LAYERS}, HiddenDim={self.config.GNN_HIDDEN_DIM}")
        print(f"NodeEmb={self.config.NODE_EMBEDDING_DIM}, EdgeEmb={self.config.EDGE_EMBEDDING_DIM}")
        print(
            f"Loss Fn: {loss_name_str}, Criterion: {type(criterion_instance).__name__}, Label Smooth: {self.config.LABEL_SMOOTHING}")
        print(f"ELR Used: {use_elr_flag}")
        if use_elr_flag:
            print(f"  ELR Beta: {self.config.ELR_BETA}, ELR NumClasses: {self.config.NUM_CLASSES}")
        print(
            f"EarlyStopping: {self.config.EARLY_STOPPING} (Patience: {self.config.EARLY_STOPPING_PATIENCE}, MinDelta: {self.config.EARLY_STOPPING_MIN_DELTA})")
        print(f"Checkpoints: {self.checkpoints_path}")
        print(f"Logs: {self.logs_path}")
        print(f"--------------------\n")

    def load(self, dataset_name):
        """
        Load model from the last checkpoint.

        Args:
            dataset_name: Name of the dataset to load checkpoint for

        Returns:
            int: Last epoch number loaded, or 0 if no checkpoint found
        """
        if not hasattr(self, 'model') or (self.model is None and self.model1 is None):
            print("Models not initialized. Call setup_model_and_trainer() first.")
            return 0

        # Find all checkpoints for this dataset
        if self.config.CO_TEACHING:
            pattern1 = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_*_model1.pth')
            pattern2 = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_*_model2.pth')
            files1 = glob.glob(pattern1)
            files2 = glob.glob(pattern2)

            if not files1 or not files2:
                print(f"No checkpoints found for dataset {dataset_name}")
                return 0

            # Extract epoch numbers and find the maximum
            epochs1 = [int(f.split('_epoch_')[1].split('_model1.pth')[0]) for f in files1]
            epochs2 = [int(f.split('_epoch_')[1].split('_model2.pth')[0]) for f in files2]

            # Find common epochs (both models must exist)
            common_epochs = set(epochs1) & set(epochs2)
            if not common_epochs:
                print(f"No matching checkpoint pairs found for dataset {dataset_name}")
                return 0

            last_epoch = max(common_epochs)

            # Load both models
            model1_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{last_epoch}_model1.pth')
            model2_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{last_epoch}_model2.pth')

            try:
                self.model1.load_state_dict(torch.load(model1_path, map_location=self.device))
                self.model2.load_state_dict(torch.load(model2_path, map_location=self.device))
                print(f"Loaded co-teaching models from epoch {last_epoch}")
                return last_epoch
            except Exception as e:
                print(f"Error loading co-teaching models: {e}")
                return 0

        else:
            pattern = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_*.pth')
            files = glob.glob(pattern)

            if not files:
                print(f"No checkpoints found for dataset {dataset_name}")
                return 0

            # Extract epoch numbers and find the maximum
            epochs = [int(f.split('_epoch_')[1].split('.pth')[0]) for f in files]
            last_epoch = max(epochs)

            # Load model
            model_path = os.path.join(self.checkpoints_path, f'model_{dataset_name}_epoch_{last_epoch}.pth')

            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded single model from epoch {last_epoch}")
                return last_epoch
            except Exception as e:
                print(f"Error loading single model: {e}")
                return 0

    def train(self, dataset_name, train_data, val_data=None, loss_fn_name="ce",
              use_cleanlab=False, start_epoch=1):
        """
        Train the model.

        Args:
            dataset_name: Name of the dataset (for checkpointing and logging)
            train_data: List of training graph data objects
            val_data: List of validation graph data objects (optional)
            loss_fn_name: Name of loss function to use
            use_cleanlab: Whether to apply cleanlab preprocessing
            start_epoch: Epoch to start training from (useful for resuming)
        """
        self.current_dataset_name = dataset_name
        self.training_logs = []  # Reset logs for new training

        if not train_data or len(train_data) == 0:
            print("Training data is empty. Aborting training.")
            return

        if use_cleanlab:
            cl_processor = CleanLabProcessor(self.config, self.device, self.model_class)
            train_data = cl_processor.apply_cleanlab_to_data(
                train_data,
                cleanlab_train_epochs=getattr(self.config, 'CLEANLAB_EPOCHS', 10),
                cleanlab_lr=getattr(self.config, 'CLEANLAB_LR', 1e-3),
                cleanlab_batch_size_multiplier=getattr(self.config, 'CLEANLAB_BS_MULTIPLIER', 1)
            )
            if not train_data or len(train_data) == 0:
                print("Training data empty after Cleanlab. Aborting.")
                return

        self._validate_data_dimensions(train_data)
        self.setup_model_and_trainer()

        # ELR specific checks
        if self.config.USE_ELR:
            if self.config.CO_TEACHING:
                print(
                    "Warning: ELR with Co-Teaching in this setup is not fully defined. CoTeachingTrainer will use base loss.")
            elif self.teacher_model is None:
                raise ValueError("Teacher model not initialized for ELR. Check setup_model_and_trainer.")
            if not hasattr(self.config, 'NUM_CLASSES') or not hasattr(self.config, 'ELR_BETA'):
                raise ValueError("config.NUM_CLASSES and config.ELR_BETA must be set for ELRLoss.")

        train_loader, val_loader = self.create_data_loaders(train_data, val_data)
        if not train_loader or (hasattr(train_loader, 'dataset') and len(train_loader.dataset) == 0):
            print("Train loader empty. Aborting.")
            return

        # --- Criterion Setup ---
        base_criterion_for_eval_or_coteaching = LossManager.get_loss(loss_fn_name,
                                                                     label_smoothing=self.config.LABEL_SMOOTHING)
        training_criterion = None

        if self.config.CO_TEACHING:
            (optimizer1, optimizer2), (scheduler1, scheduler2) = self.create_optimizers_and_schedulers()
            training_criterion = base_criterion_for_eval_or_coteaching
        else:
            optimizer, scheduler = self.create_optimizers_and_schedulers()
            base_criterion_for_single_train = LossManager.get_loss(loss_fn_name,
                                                                   label_smoothing=self.config.LABEL_SMOOTHING)
            if self.config.USE_ELR:
                training_criterion = ELRLoss(teacher_model=self.teacher_model,
                                             criterion=base_criterion_for_single_train,
                                             )
            else:
                training_criterion = base_criterion_for_single_train

        early_stopping = None
        if self.config.EARLY_STOPPING and val_loader:
            early_stopping = EarlyStopping(patience=self.config.EARLY_STOPPING_PATIENCE,
                                           min_delta=self.config.EARLY_STOPPING_MIN_DELTA, mode='max')

        self._print_config(dataset_name, loss_fn_name, training_criterion, self.config.USE_ELR)

        print("Starting training...")
        best_val_acc = 0.0

        for epoch in range(start_epoch, self.config.EPOCHS + 1):
            start_time = time.time()
            val_acc_epoch = 0.0
            epoch_metrics = {}

            if self.config.CO_TEACHING:
                train_loss1, train_loss2, forget_rate = self.trainer.train_epoch(
                    train_loader, optimizer1, optimizer2, training_criterion
                )
                val_loss1_disp, val_loss2_disp, val_acc1, val_acc2 = 0.0, 0.0, 0.0, 0.0
                if val_loader:
                    (v_loss1, v_loss2), (v_acc1, v_acc2) = self.trainer.eval_epoch(val_loader,
                                                                                   base_criterion_for_eval_or_coteaching)
                    val_loss1_disp, val_loss2_disp, val_acc1, val_acc2 = v_loss1, v_loss2, v_acc1, v_acc2
                    val_acc_epoch = max(val_acc1, val_acc2)

                    if val_acc_epoch > best_val_acc:
                        best_val_acc = val_acc_epoch

                cur_lr = optimizer1.param_groups[0]['lr']

                epoch_metrics = {
                    'forget_rate': forget_rate,
                    'train_loss1': train_loss1,
                    'train_loss2': train_loss2,
                    'val_loss1': val_loss1_disp,
                    'val_loss2': val_loss2_disp,
                    'val_acc1': val_acc1,
                    'val_acc2': val_acc2,
                    'learning_rate': cur_lr,
                    'best_val_acc': best_val_acc
                }

                print(
                    f"Ep {epoch:03d}/{self.config.EPOCHS:03d} | FR: {forget_rate:.3f} | TrL1: {train_loss1:.4f}, TrL2: {train_loss2:.4f} | "
                    f"VaL1: {val_loss1_disp:.4f}, VaL2: {val_loss2_disp:.4f} | VaAcc1: {val_acc1:.4f}, VaAcc2: {val_acc2:.4f} | "
                    f"LR: {cur_lr:.1e} | Time: {(time.time() - start_time):.2f}s")

                if scheduler1: scheduler1.step()
                if scheduler2: scheduler2.step()

            else:  # Single Model
                teacher_for_train = self.teacher_model if self.config.USE_ELR else None
                train_loss = self.trainer.train_epoch(train_loader, optimizer, training_criterion, teacher_for_train)

                val_loss_disp, val_acc_disp = 0.0, 0.0
                if val_loader:
                    val_loss, val_acc = self.trainer.eval_epoch(val_loader, training_criterion)
                    val_loss_disp, val_acc_disp = val_loss, val_acc
                    val_acc_epoch = val_acc

                    if val_acc_epoch > best_val_acc:
                        print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc_epoch:.4f}. Saving checkpoint...")
                        self._save_checkpoint(epoch, dataset_name)
                        best_val_acc = val_acc_epoch

                cur_lr = optimizer.param_groups[0]['lr']

                epoch_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss_disp,
                    'val_acc': val_acc_disp,
                    'learning_rate': cur_lr,
                    'best_val_acc': best_val_acc
                }

                print(
                    f"Ep {epoch:03d}/{self.config.EPOCHS:03d} | TrLoss: {train_loss:.4f} | VaLoss: {val_loss_disp:.4f} | "
                    f"VaAcc: {val_acc_disp:.4f} | LR: {cur_lr:.1e} | Time: {(time.time() - start_time):.2f}s")

                if scheduler: scheduler.step()

            # Save checkpoint when validation accuracy improves


            # Log epoch information
            elapsed_time = time.time() - start_time
            self._log_epoch(epoch, epoch_metrics, elapsed_time)

            if early_stopping and val_loader:
                if early_stopping(val_acc_epoch):
                    print(
                        f"Early stopping at epoch {epoch}. Best val_acc: {best_val_acc:.4f} (from ep {epoch - early_stopping.counter})")
                    break

        print("\nTraining finished.")

        # Save final training log
        self._save_training_log(dataset_name)
        print(f"Training logs saved to {os.path.join(self.logs_path, f'training_log_{dataset_name}.json')}")

    def generate_test_predictions(self, test_dataset, dataset_name, output_path, loss_fn_name="ce"):
        """
        Generate predictions on a single test dataset.

        Args:
            test_dataset: List of test graph data objects
            dataset_name: Name of the dataset (used for loading the appropriate model and naming the output file)
            output_path: Directory path where the prediction CSV file will be saved
            loss_fn_name: Loss function name for evaluation criterion
        """
        print(f"\n--- Generating Test Predictions for dataset: {dataset_name} ---")

        if not test_dataset or len(test_dataset) == 0:
            print("Test data is empty. Skipping prediction.")
            return

        os.makedirs(output_path, exist_ok=True)

        # Ensure models are available
        if not ((self.config.CO_TEACHING and self.model1 and self.model2) or
                (not self.config.CO_TEACHING and self.model)):
            print("Models not available. Make sure to train or load models first.")
            return

        # Create evaluation criterion
        criterion_for_eval = LossManager.get_loss(
            loss_fn_name,
            label_smoothing=getattr(self.config, 'LABEL_SMOOTHING', 0.0)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=getattr(self.config, 'NUM_WORKERS', 0)
        )

        preds_array = None

        if self.config.CO_TEACHING:
            if not self.trainer or not hasattr(self.trainer, 'eval_epoch'):
                print("CoTeachingTrainer or its eval_epoch not available.")
                return
            preds1, preds2 = self.trainer.eval_epoch(test_loader, criterion_for_eval, is_test_set_preds_only=True)
            preds_array = preds1 if preds1 else preds2
            if preds_array is None or len(preds_array) == 0:
                print("No predictions generated by either model.")
                return
        else:
            if not self.trainer or not hasattr(self.trainer, 'eval_epoch'):
                print("SingleModelTrainer or its eval_epoch not available.")
                return
            preds_array = self.trainer.eval_epoch(test_loader, criterion_for_eval, is_test_set_preds_only=True)
            if preds_array is None or len(preds_array) == 0:
                print("No predictions generated.")
                return

        num_test_samples = len(preds_array)
        ids = np.arange(1, num_test_samples + 1)
        pred_df = pd.DataFrame({'id': ids, 'pred': preds_array})
        output_fn = os.path.join(output_path, f'testset_{dataset_name}.csv')
        pred_df.to_csv(output_fn, index=False)
        print(f"Test predictions for {dataset_name} ({num_test_samples} samples) saved to {output_fn}")
        print("---------------------------------")
