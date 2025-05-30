# cleanlab_processor.py
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from cleanlab.filter import find_label_issues

from trainers import SingleModelTrainer  # To train a temporary model


# from networks import MyLocalGatedGCN # This will be passed as model_class

class CleanLabProcessor:
    def __init__(self, main_config, device, model_class):
        self.main_config = main_config
        self.device = device
        self.model_class = model_class  # e.g., MyLocalGatedGCN

    def apply_cleanlab_to_data(self, train_data_list,
                               cleanlab_train_epochs=10,
                               cleanlab_lr=1e-3,
                               cleanlab_batch_size_multiplier=1):
        """
        Applies Cleanlab to the training data to identify and potentially remove noisy labels.
        Returns a filtered list of training data.
        """
        print("\n--- Applying Cleanlab Preprocessing ---")
        if not train_data_list or len(train_data_list) == 0:
            print("No training data to apply Cleanlab to. Skipping.")
            return train_data_list

        # 1. Temporary model setup
        print("Setting up temporary model for Cleanlab...")
        temp_model_config = copy.deepcopy(self.main_config)
        temp_model_config.USE_ELR = False
        temp_model_config.CO_TEACHING = False

        temp_model = self.model_class(temp_model_config).to(self.device)
        temp_optimizer = torch.optim.AdamW(temp_model.parameters(), lr=cleanlab_lr,
                                           weight_decay=self.main_config.WEIGHT_DECAY)
        temp_criterion = nn.CrossEntropyLoss(label_smoothing=self.main_config.LABEL_SMOOTHING)
        temp_trainer = SingleModelTrainer(temp_model, temp_model_config, self.device)

        # 2. DataLoader for current training data (shuffle=False is critical)
        effective_bs = max(1, int(self.main_config.BATCH_SIZE * cleanlab_batch_size_multiplier))
        temp_train_loader = DataLoader(train_data_list, batch_size=effective_bs, shuffle=False, num_workers=0)

        # 3. Train temporary model
        print(f"Training temporary model for {cleanlab_train_epochs} epochs (LR={cleanlab_lr}, BS={effective_bs})...")
        for epoch in range(1, cleanlab_train_epochs + 1):
            epoch_loss = temp_trainer.train_epoch(temp_train_loader, temp_optimizer, temp_criterion)
            print(f"Cleanlab pre-train Epoch {epoch}/{cleanlab_train_epochs}, Loss: {epoch_loss:.4f}")

        # 4. Get predicted probabilities
        print("Getting predicted probabilities for Cleanlab...")
        # SingleModelTrainer.eval_epoch returns (labels, probs) when get_probs=True
        given_labels, pred_probs = temp_trainer.eval_epoch(temp_train_loader, temp_criterion, get_probs=True)

        if not isinstance(pred_probs, np.ndarray) or pred_probs.ndim == 1 or pred_probs.shape[0] == 0:
            print(
                f"Error: Pred probs array malformed (shape: {pred_probs.shape if isinstance(pred_probs, np.ndarray) else 'N/A'}). Cleanlab aborted.")
            return train_data_list
        if pred_probs.shape[0] != len(train_data_list):
            print(
                f"Error: Num predictions ({pred_probs.shape[0]}) != train set size ({len(train_data_list)}). Cleanlab aborted.")
            return train_data_list

        if given_labels.ndim > 1 and given_labels.shape[1] == 1: given_labels = given_labels.squeeze()
        try:
            given_labels = given_labels.astype(int)
        except ValueError as e:
            print(f"Error converting labels to int: {e}. Cleanlab aborted.")
            return train_data_list

        num_original_samples = len(train_data_list)
        valid_indices_for_cl = (given_labels != -1)

        if not np.all(valid_indices_for_cl):
            print(f"Excluding {np.sum(~valid_indices_for_cl)} samples with label -1 from Cleanlab analysis.")
            pred_probs_for_cl = pred_probs[valid_indices_for_cl]
            given_labels_for_cl = given_labels[valid_indices_for_cl]
            original_indices_map = np.where(valid_indices_for_cl)[0]
        else:
            pred_probs_for_cl = pred_probs
            given_labels_for_cl = given_labels
            original_indices_map = np.arange(num_original_samples)

        if len(pred_probs_for_cl) == 0:
            print("No valid samples (labels != -1) for Cleanlab. Skipping noise detection.")
        else:
            print("Running Cleanlab to find label issues on valid samples...")
            try:
                num_classes_from_probs = pred_probs_for_cl.shape[1]
                min_label, max_label = np.min(given_labels_for_cl), np.max(given_labels_for_cl)

                if not (0 <= min_label <= max_label < num_classes_from_probs):
                    print(f"Error: Labels for Cleanlab outside expected range [0, {num_classes_from_probs - 1}]. "
                          f"Min: {min_label}, Max: {max_label}. Skipping Cleanlab on this subset.")
                    issue_indices_subset = np.array([], dtype=int)
                else:
                    issue_indices_subset = find_label_issues(
                        labels=given_labels_for_cl,
                        pred_probs=pred_probs_for_cl,
                        return_indices_ranked_by='self_confidence'
                    )
            except Exception as e:
                print(f"Error during find_label_issues: {e}. Skipping Cleanlab noise removal.")
                print(f"Debug: pred_probs_for_cl shape: {pred_probs_for_cl.shape}, "
                      f"labels shape: {given_labels_for_cl.shape}, dtype: {given_labels_for_cl.dtype}, "
                      f"min/max: {np.min(given_labels_for_cl)}/{np.max(given_labels_for_cl)}")
                issue_indices_subset = np.array([], dtype=int)

            if issue_indices_subset is None or len(issue_indices_subset) == 0:
                print("Cleanlab found no label issues among valid samples.")
            else:
                print(f"Cleanlab ID'd {len(issue_indices_subset)} issues from {len(given_labels_for_cl)} samples.")
                noisy_indices_original = original_indices_map[issue_indices_subset]

                keep_mask = np.ones(num_original_samples, dtype=bool)
                keep_mask[noisy_indices_original] = False

                filtered_train_data = [sample for i, sample in enumerate(train_data_list) if keep_mask[i]]
                num_removed = num_original_samples - len(filtered_train_data)
                print(
                    f"Removed {num_removed} noisy samples. Train size: {num_original_samples} -> {len(filtered_train_data)}.")
                train_data_list = filtered_train_data

        del temp_model, temp_optimizer, temp_criterion, temp_trainer, temp_train_loader
        del pred_probs, given_labels
        if 'pred_probs_for_cl' in locals(): del pred_probs_for_cl
        if 'given_labels_for_cl' in locals(): del given_labels_for_cl
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("--- Cleanlab Preprocessing Finished ---")
        return train_data_list
