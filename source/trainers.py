# trainers.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.data import Batch

from losses import ELRLoss
from training_utils import drop_edges  # Import from our new utils file


class SingleModelTrainer:
    """Handles training and evaluation logic for single model"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def train_epoch(self, loader, optimizer, criterion, teacher_model_for_elr_update=None):
        self.model.train()
        total_loss, processed_graphs = 0, 0
        for data in loader:
            data = data.to(self.device)
            data = drop_edges(data, self.config.EDGE_DROPPING)

            optimizer.zero_grad()
            out = self.model(data)
            target_y = data.y.squeeze()
            if target_y.ndim == 0:
                target_y = target_y.unsqueeze(0)

            # ELRLoss might need the `data` object if the teacher model needs to make predictions on it
            # or if the loss calculation itself depends on more than just `out` and `target_y`.
            if self.config.USE_ELR and isinstance(criterion, ELRLoss):  # ELRLoss is a custom class from losses.py
                loss = criterion(out, target_y, data)  # Pass data for ELRLoss
            else:
                loss = criterion(out, target_y)

            loss.backward()
            optimizer.step()

            # If ELR is used, the teacher model (which is part of ELRLoss criterion)
            # needs to be updated with the student model's weights (EMA update typically)
            # The original code had `teacher_model.load_state_dict(self.model.state_dict())`
            # This implies a direct copy, not EMA. This logic should be tied to ELRLoss or handled carefully.
            # For now, we assume ELRLoss handles its teacher_model internally, or GatedGCNTrainingSystem does.
            # The provided `ELRLoss` definition implies the teacher model is passed at init.
            # The update of teacher model's parameters (e.g., EMA) should happen *after* student's optimizer.step().
            # If `teacher_model_for_elr_update` is the actual teacher model instance (not ELRLoss wrapper):
            if teacher_model_for_elr_update:
                # This is likely an EMA update in a real ELR scenario, not direct copy after every batch.
                # For simplicity, sticking to the original direct copy:
                teacher_model_for_elr_update.load_state_dict(self.model.state_dict())

            total_loss += loss.item() * data.num_graphs
            processed_graphs += data.num_graphs
        return total_loss / processed_graphs if processed_graphs else 0

    @torch.no_grad()
    def eval_epoch(self, loader, criterion, is_test_set_preds_only=False, get_probs=False):
        self.model.eval()
        total_loss, processed_graphs = 0, 0
        all_preds_list, all_labels_list, all_probs_list = [], [], []
        use_elr = self.config.USE_ELR

        for data_batch in loader:
            data_batch = data_batch.to(self.device)
            out = self.model(data_batch)

            target_y_cpu = None
            if hasattr(data_batch, 'y') and data_batch.y is not None:
                target_y_cpu = data_batch.y.squeeze().cpu()
                if target_y_cpu.ndim == 0:
                    target_y_cpu = target_y_cpu.unsqueeze(0)

            if target_y_cpu is not None:
                all_labels_list.append(target_y_cpu)

            if get_probs:
                if out.ndim == 2 and out.shape[1] == 1:  # Binary with single logit
                    prob_class_1 = torch.sigmoid(out.cpu())
                    prob_class_0 = 1.0 - prob_class_1
                    all_probs_list.append(torch.cat((prob_class_0, prob_class_1), dim=1))
                else:  # Multi-class or binary with two logits
                    all_probs_list.append(F.softmax(out.cpu(), dim=1))

            elif is_test_set_preds_only:
                preds = out.argmax(dim=1)
                all_preds_list.append(preds.cpu())
            else:
                preds = out.argmax(dim=1)
                all_preds_list.append(preds.cpu())

                if target_y_cpu is not None:
                    valid_targets = target_y_cpu != -1
                    if valid_targets.any():
                        if use_elr and isinstance(criterion, ELRLoss):
                            loss = criterion(out[valid_targets.to(self.device)],
                                             target_y_cpu[valid_targets].to(self.device),
                                             data_batch)
                        else:
                            loss = criterion(out[valid_targets.to(self.device)],
                                             target_y_cpu[valid_targets].to(self.device))
                        total_loss += loss.item() * torch.sum(valid_targets).item()

            processed_graphs += data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else data_batch.x.size(0)

        if get_probs:
            # Concatenate labels and probs
            final_probs = torch.cat(all_probs_list).numpy() if all_probs_list else np.array([])
            final_labels = torch.cat(all_labels_list).numpy() if all_labels_list else np.array([])

            # Filter out invalid labels
            valid_indices = final_labels != -1
            if np.any(valid_indices):
                return final_labels[valid_indices], final_probs[valid_indices]
            return np.array([]), np.array([])

        if is_test_set_preds_only:
            return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])

        if not all_labels_list:
            return 0.0, 0.0

        all_preds_np = torch.cat(all_preds_list).numpy()
        all_labels_np = torch.cat(all_labels_list).numpy()

        valid_indices = all_labels_np != -1
        accuracy = 0.0
        num_valid_targets = np.sum(valid_indices)
        if num_valid_targets > 0:
            accuracy = accuracy_score(all_labels_np[valid_indices], all_preds_np[valid_indices])

        effective_loss = total_loss / num_valid_targets if num_valid_targets > 0 else 0.0
        return effective_loss, accuracy


class CoTeachingTrainer:
    def __init__(self, model1, model2, config, device):
        self.model1 = model1
        self.model2 = model2
        self.config = config
        self.device = device
        self.epoch = 0
        self.num_gradual = config.NUM_GRADUAL
        self.exponent = config.EXPONENT
        self.max_forget_rate = config.MAX_CO_TEACHING_FORGET_RATE
        self.forget_rate_max_epoch = config.FORGET_RATE_MAX_EPOCH

        # Internal single model trainers for evaluation logic if needed, or replicate eval logic
        # This simplifies eval calls; assumes SingleModelTrainer can eval a given model.
        self._eval_helper_m1 = SingleModelTrainer(self.model1, config, device)
        self._eval_helper_m2 = SingleModelTrainer(self.model2, config, device)

    def _get_current_forget_rate(self):
        if self.epoch < self.num_gradual:
            return 0.0
        else:
            progress = (self.epoch - self.num_gradual) / \
                       max(1, (self.forget_rate_max_epoch - self.num_gradual))  # Avoid division by zero
            rate = self.max_forget_rate * (progress ** self.exponent)
            return min(self.max_forget_rate, rate)

    def _select_samples_by_loss(self, data_batch, selecting_model, forget_rate):
        selecting_model.eval()
        with torch.no_grad():
            outputs = selecting_model(data_batch)
            targets = data_batch.y.squeeze()
            if targets.ndim == 0: targets = targets.unsqueeze(0)

            # Handle cases where batch size might be 1 after filtering
            if outputs.shape[0] == 0:  # No samples to select from
                return np.array([], dtype=int)

            losses = F.cross_entropy(outputs, targets, reduction='none')
            num_remember = max(1, int((1 - forget_rate) * len(losses)))
            if len(losses) == 0:  # Should not happen if outputs.shape[0] > 0
                return np.array([], dtype=int)
            _, clean_indices = torch.topk(-losses, min(num_remember, len(losses)))  # Ensure k <= len(losses)
        return clean_indices.cpu().numpy()

    def train_epoch(self, loader, optimizer1, optimizer2, criterion):
        self.model1.train()
        self.model2.train()
        total_loss1, total_loss2 = 0, 0
        processed_graphs = 0
        current_forget_rate = self._get_current_forget_rate()

        for batch_data in loader:
            batch_data = batch_data.to(self.device)
            batch_data_dropped_edges = drop_edges(batch_data, self.config.EDGE_DROPPING)

            # Use the batch with dropped edges for selection and training
            data_list = batch_data_dropped_edges.to_data_list()
            batch_size = len(data_list)  # batch_data_dropped_edges.num_graphs

            if batch_size == 0: continue

            if batch_size == 1:
                clean_indices_for_model1 = [0]
                clean_indices_for_model2 = [0]
            else:
                clean_indices_for_model1 = self._select_samples_by_loss(
                    batch_data_dropped_edges, self.model2, current_forget_rate
                )
                clean_indices_for_model2 = self._select_samples_by_loss(
                    batch_data_dropped_edges, self.model1, current_forget_rate
                )

            if len(clean_indices_for_model1) > 0:
                clean_data_for_model1 = Batch.from_data_list([data_list[i] for i in clean_indices_for_model1]).to(
                    self.device)
                optimizer1.zero_grad()
                outputs1 = self.model1(clean_data_for_model1)
                targets1 = clean_data_for_model1.y.squeeze()
                if targets1.ndim == 0: targets1 = targets1.unsqueeze(0)
                loss1 = criterion(outputs1, targets1)
                loss1.backward()
                optimizer1.step()
                total_loss1 += loss1.item() * len(clean_indices_for_model1)

            if len(clean_indices_for_model2) > 0:
                clean_data_for_model2 = Batch.from_data_list([data_list[i] for i in clean_indices_for_model2]).to(
                    self.device)
                optimizer2.zero_grad()
                outputs2 = self.model2(clean_data_for_model2)
                targets2 = clean_data_for_model2.y.squeeze()
                if targets2.ndim == 0: targets2 = targets2.unsqueeze(0)
                loss2 = criterion(outputs2, targets2)
                loss2.backward()
                optimizer2.step()
                total_loss2 += loss2.item() * len(clean_indices_for_model2)

            processed_graphs += batch_size

        self.epoch += 1
        avg_loss1 = total_loss1 / processed_graphs if processed_graphs > 0 else 0
        avg_loss2 = total_loss2 / processed_graphs if processed_graphs > 0 else 0
        return avg_loss1, avg_loss2, current_forget_rate

    @torch.no_grad()
    def eval_epoch(self, loader, criterion, is_test_set_preds_only=False):
        # Note: CoTeaching original eval_epoch was incomplete. This version evaluates both models.
        # `get_probs` functionality could be added if needed for co-teaching eval.

        # Eval model1
        loss1, acc1 = self._eval_helper_m1.eval_epoch(loader, criterion, is_test_set_preds_only)

        # Eval model2
        loss2, acc2 = self._eval_helper_m2.eval_epoch(loader, criterion, is_test_set_preds_only)

        if is_test_set_preds_only:
            # loss1 and loss2 here are actually predictions array from SingleModelTrainer's eval_epoch
            return loss1, loss2  # Returning (preds1, preds2)

        return (loss1, loss2), (acc1, acc2)  # Returning (losses_tuple, accuracies_tuple)