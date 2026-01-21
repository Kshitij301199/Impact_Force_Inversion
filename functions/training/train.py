import os
import sys
import copy
import json
import torch
import numpy as np
import torch.nn as nn
torch.set_default_dtype(torch.float32)
import torch.optim as optim
import matplotlib.pyplot as plt

with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
    paths = json.load(file)
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
    data_params = json.load(file)

sys.path.append(paths['BASE_DIR'])
from functions.utils import *

def set_seed(seed=42):
    # torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # disable benchmarking for reproducibility

def grad_stats(model):
    norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    return np.mean(norms).item(), np.max(norms).item()

class ModelTrainer:
    """
    A class to handle the training, validation, and testing of a PyTorch model.
    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        warmup_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for warm-up phase.
        main_scheduler (torch.optim.lr_scheduler._LRScheduler): Main learning rate scheduler.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        model_dir (str): Directory to save the model.
        curve_file (str): File to log training curves.
        interval (str, optional): Interval identifier for the model. Defaults to None.
        test_julday (str, optional): Test julian day identifier. Defaults to None.
        val_julday (str, optional): Validation julian day identifier. Defaults to None.
        model_type (str, optional): Type of the model. Defaults to "Model".
        device (torch.device, optional): Device to run the model on. Defaults to CUDA if available.
        monitor1 (torch.nn.Module, optional): First monitoring metric. Defaults to None.
        monitor2 (torch.nn.Module, optional): Second monitoring metric. Defaults to None.
    Methods:
        train(num_epochs, patience): Trains the model for a specified number of epochs with early stopping.
        check_train(mult_by=50): Evaluates the model on the training set.
        test(mult_by=50): Evaluates the model on the test set.
        _evaluate(mult_by, dataloader, save_path): Helper method to evaluate the model on a given DataLoader.
        _run_epoch(dataloader, training=False): Helper method to run a single epoch of training or validation.
    """
    def __init__(self, model, criterion, optimizer, warmup_scheduler, main_scheduler,
                 train_loader, val_loader, test_loader, model_dir, curve_file, 
                 interval=None, test_julday=None, val_julday=None, model_type="Model", device=None,
                 monitor1=None, monitor2=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_dir = model_dir
        self.curve_file = curve_file
        self.interval = interval
        self.test_julday = test_julday
        self.val_julday = val_julday
        self.model_type = model_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.monitor1 = monitor1
        self.monitor2 = monitor2

        self.model.to(self.device)
        set_seed()

    def train(self, num_epochs, patience):
        best_loss = float('inf')
        best_epoch = 0
        best_weights = None
        consecutive_increase = 0

        start_time = get_current_time()

        for epoch in range(num_epochs):
            if self.monitor1 is not None:
                train_loss, train_mse, train_wmse, mean_gs, max_gs = self._run_epoch(self.train_loader, training=True)
                val_loss, val_mse, val_wmse, val_mean_gs, val_max_gs = self._run_epoch(self.val_loader, training=False)
            else:
                train_loss, mean_gs, max_gs = self._run_epoch(self.train_loader, training=True)
                val_loss, val_mean_gs, val_max_gs = self._run_epoch(self.val_loader, training=False)

            with open(self.curve_file, "a") as file:
                file.write(f"{epoch};{train_loss:.5f};{val_loss:.5f};{self.optimizer.param_groups[0]['lr']:.2e};{mean_gs};{max_gs}\n")
            
            if self.warmup_scheduler is None:
                self.main_scheduler.step(val_loss)
                print(f"Epoch {epoch+1} Plateau LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            else:
                if epoch < 10:
                    self.warmup_scheduler.step()
                    print(f"Epoch {epoch+1} Warmup LR: {self.warmup_scheduler.get_last_lr()[0]:.2e}")
                else:
                    self.main_scheduler.step(val_loss)
                    print(f"Epoch {epoch+1} Plateau LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if self.monitor1 is not None:
                print(f"\t\tMonitoring -- MSE : {val_mse:.4f}, Weighted MSE : {val_wmse:.4f}")

            if epoch < 50:
                tolerance = 0.05  # acceptable deviation, e.g., 0.5% or absolute value (adjust as needed)
            elif (epoch < 100) & (epoch >= 50):
                tolerance = 0.01
            else:
                tolerance = 0.005 

            if val_loss < best_loss:
                # not strictly better
                best_loss = val_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                model_path = f"{self.model_dir}/t{self.test_julday}_v{self.val_julday}_{self.interval}_{self.model_type}_model.pt"
                torch.save(best_weights, model_path)
                print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
                consecutive_increase = 0

            elif val_loss <= best_loss + (best_loss * tolerance):
                # within tolerance range, don't penalize
                print(f"Epoch {epoch + 1}: Validation loss {val_loss:.4f} within tolerance of best ({best_loss:.4f}).")
                # reset patience counter or leave unchanged, depending on your preference
                # consecutive_increase = 0

            else:
                # worse beyond tolerance
                consecutive_increase += 1
                print(f"Epoch {epoch + 1}: Validation loss increased ({val_loss:.4f}). Patience {consecutive_increase}/{patience}")

            if consecutive_increase > patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1} with val loss {best_loss:.4f}")
                break

        self.model.load_state_dict(best_weights)
        end_time = get_current_time()
        self.time_to_train = get_time_elapsed(start_time, end_time)

    def _run_epoch(self, dataloader, training=False):
        epoch_loss = 0.0
        if self.monitor1 is not None:
            epoch_mse = 0.0
            epoch_wmse = 0.0
        self.model.train() if training else self.model.eval()
        mean_gs, max_gs = [], []
        for i, (input_sequences, target_value, _) in enumerate(dataloader):
            if input_sequences.dim() == 2:
                continue
            # if self.model_type == "LinReg":
            #     input_sequences = input_sequences.squeeze(1)
            input_sequences = input_sequences.float().to(self.device)
            target_value = target_value.float().to(self.device)

            if training:
                self.optimizer.zero_grad()

            output = self.model(input_sequences).squeeze(1)

            loss = self.criterion(output, target_value)
            if self.monitor1 is not None:
                epoch_mse += self.monitor1(output, target_value).item()
                epoch_wmse += self.monitor2(output, target_value).item()

            if training:
                loss.backward()
                self.optimizer.step()
                if i % 5 == 0:  # every 5 steps
                    mean_g, max_g = grad_stats(self.model)
                    mean_gs.append(mean_g)
                    max_gs.append(max_g)
                    
            epoch_loss += loss.item()
        if self.monitor1 is not None:
            return epoch_loss / len(dataloader), epoch_mse / len(dataloader), epoch_wmse / len(dataloader), mean_gs, max_gs
        else:
            return epoch_loss / len(dataloader), mean_gs, max_gs

    def check_train(self, mult_by=50):
        return self._evaluate(mult_by, self.train_loader, save_path=f"{self.model_dir}/t{self.test_julday}_v{self.val_julday}_{self.interval}_{self.model_type}_model.pt")
    
    def test(self, mult_by=50):
        return self._evaluate(mult_by, self.test_loader, save_path=f"{self.model_dir}/t{self.test_julday}_v{self.val_julday}_{self.interval}_{self.model_type}_model.pt")

    def _evaluate(self, mult_by, dataloader, save_path):
        self.model.load_state_dict(torch.load(save_path, weights_only=True, map_location=self.device))
        self.model.eval()

        in_seq, preds, targets, timestamps = [], [], [], []
        total_loss = 0.0
        if self.monitor1 is not None:
            total_mse = 0.0
            total_wmse = 0.0

        with torch.no_grad():
            for input_sequences, target_value, ts in dataloader:
                if input_sequences.dim() == 2:
                    continue
                # if self.model_type == "LinReg":
                #     input_sequences = input_sequences.squeeze(1)
                input_sequences = input_sequences.float().to(self.device)
                target_value = target_value.float().to(self.device)

                output = self.model(input_sequences).squeeze(1)

                loss = self.criterion(output, target_value)
                total_loss += loss.item()
                if self.monitor1 is not None:
                    total_mse += self.monitor1(output, target_value)
                    total_wmse += self.monitor2(output, target_value)

                pred_unscaled = output.cpu().numpy() * mult_by
                target_unscaled = target_value.cpu().numpy() * mult_by

                in_seq.append(input_sequences.cpu().numpy())
                preds.append(pred_unscaled)
                targets.append(target_unscaled)
                timestamps.append(ts)

        print(f"Test Loss: {total_loss / len(dataloader):.4f}")
        if self.monitor1 is not None:
            print(f"\t\tMonitoring -- MSE : {total_mse / len(dataloader):.4f}, Weighted MSE : {total_wmse / len(dataloader):.4f}")
        return in_seq, preds, targets, timestamps, str(timedelta(seconds=self.time_to_train))

