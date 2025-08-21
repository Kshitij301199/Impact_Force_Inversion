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

# def train_model(model, criterion, optimizer, num_epochs:int, patience:int,
#                 interval:int, test_julday:int, val_julday:int ,model_type:str,
#                 train_dataloader, val_dataloader, test_dataloader, model_dir:str, 
#                 warmup_scheduler, main_scheduler) -> tuple[list, list, list, list]:
#     print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")
#     start_time = get_current_time()
#     print(f"{'Starting Training':-^50}")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     best_loss = float('inf')
#     best_epoch = 0
#     consecutive_increase = 0
#     best_weights = None
    
#     set_seed(42)  # Set seed for reproducibility
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         model.train()
#         # prev_target = torch.zeros(get_batch_size(interval))
#         for input_sequences, target_value, _ in train_dataloader:
#             if input_sequences.dim() == 2:
#                 continue
#             model.to(device)
#             input_sequences = input_sequences.float().to(device)  
#             # prev_target = prev_target.float().to(device)
#             target_value = target_value.float().to(device)        
            
#             optimizer.zero_grad()
#             output = model(input_sequences).squeeze(1)  # Forward pass and shape adjustment
#             assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
#             loss = criterion(output, target_value)     # Compute loss
#             loss.backward()
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             epoch_loss += loss.item()
#             # prev_target = output.cpu().detach()
#         epoch_loss /= len(train_dataloader)  # Average loss for the epoch
#         model.eval()
#         val_epoch_loss = 0.0
#         with torch.no_grad():
#             # prev_target = torch.zeros(get_batch_size(interval))
#             for input_sequences, target_value, _ in val_dataloader:
#                 if input_sequences.dim() == 2:
#                     continue
#                 model.to(device)
#                 input_sequences = input_sequences.float().to(device) 
#                 # prev_target = prev_target.float().to(device)
#                 target_value = target_value.float().to(device)
#                 # Forward pass: Get model predictions
#                 output = model(input_sequences).squeeze(1)  
#                 assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
#                 loss = criterion(output, target_value) 
#                 val_epoch_loss += loss.item()
#                 # prev_target = output.cpu().detach()
#         val_epoch_loss /= len(val_dataloader)
#         # current_lr = scheduler.optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val loss : {val_epoch_loss:.4f}, Best Loss : {best_loss:.4f}, RAM usage = {get_memory_usage_in_gb():.2f} GB")
#         # scheduler.step(val_epoch_loss)
#         if epoch < 5:
#             warmup_scheduler.step()
#             print(f"Epoch {epoch+1} Warmup LR: {warmup_scheduler.get_last_lr()[0]:.2e}")
#         else:
#             main_scheduler.step(val_epoch_loss)
#             print(f"Epoch {epoch+1} Plateau LR: {optimizer.param_groups[0]['lr']:.2e}")
#         # Check for early stopping criteria
#         if val_epoch_loss < best_loss:
#             best_loss = val_epoch_loss
#             best_epoch = epoch
#             best_weights = copy.deepcopy(model.state_dict())  # Save the best weights
#             torch.save(best_weights, f"{model_dir}/t{test_julday}_v{val_julday}_{interval}_{model_type}_model.pt")
#             print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
#             consecutive_increase = 0  # Reset counter
#         else:
#             consecutive_increase += 1  # Increment counter

#         # Stop training if loss increased for more than early_stop_patience epochs
#         if consecutive_increase > patience:
#             torch.save(best_weights, f"{model_dir}/t{test_julday}_v{val_julday}_{interval}_{model_type}_model.pt")
#             with open(f"{model_dir}/best_epoch.txt", 'a') as f:
#                 f.write(f"File: t{test_julday}_v{val_julday}_{interval}_{model_type}, Best epoch: {best_epoch + 1}, Best loss: {best_loss:.4f}\n")
#             print(f"Early stopping triggered. Restoring model to epoch {best_epoch + 1} with lowest loss {best_loss:.4f}")
#             break
    
#     model.load_state_dict(best_weights)
#     # torch.save(model.state_dict(), f"{model_dir}/t{test_julday}_v{val_julday}_{interval}_{model_type}_model.pt")
#     end_time = get_current_time()
#     time_to_train = get_time_elapsed(start_time, end_time)
#     print(f"{'End Training':-^50}")

#     # Test model
#     start_time = get_current_time()
#     print(f"{'Start Testing':-^50}")
#     model.eval()
#     in_sequence, predicted_output, target_output, timestamps = [], [], [], []
#     test_epoch_loss = 0.0
#     with torch.no_grad():
#         # prev_target = torch.zeros(get_batch_size(interval))
#         for input_sequences, target_value, test_timestamps in test_dataloader:
#             if input_sequences.dim() == 2:
#                 continue
#             model.to(device)
#             # Move data to the appropriate device if using a GPU
#             input_sequences = input_sequences.float().to(device)  # Shape: (batch_size, 20, 3000)
#             # prev_target = prev_target.float().to(device)
#             target_value = target_value.float().to(device)        # Shape: (batch_size,)

#             # Forward pass: Get model predictions
#             output = model(input_sequences).squeeze(1)  # Shape: (batch_size, 1)
#             assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
#             loss = criterion(output, target_value) 
#             test_epoch_loss += loss.item()
#             # prev_target = output.cpu().detach()
#             # Squeeze the output to match target shape
#             if scaler is None:
#                 in_sequence.append(input_sequences.cpu().numpy())
#                 predicted_output.append(output.detach().cpu().numpy())
#                 target_output.append(target_value.cpu().numpy())
#                 timestamps.append(test_timestamps)
#             else:
#                 in_sequence.append(input_sequences.cpu().numpy())
#                 rem = output.cpu().detach().cpu().numpy().shape
#                 predicted_output.append(scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1,1)).reshape(rem))
#                 target_output.append(scaler.inverse_transform(target_value.cpu().numpy().reshape(-1,1)).reshape(rem))
#                 timestamps.append(test_timestamps)
#     print(f"Test loss : {test_epoch_loss / len(test_dataloader)}")
#     end_time = get_current_time()
#     time_to_test = get_time_elapsed(start_time, end_time)
#     print(f"{'End Testing':-^50}")

#     # Sanity check
#     start_time = get_current_time()
#     print(f"{'Start Testing':-^50}")
#     model.load_state_dict(torch.load(f"{model_dir}/t{test_julday}_v{val_julday}_{interval}_{model_type}_model.pt", weights_only=True))
#     model.eval()
#     in_sequence, predicted_output, target_output, timestamps = [], [], [], []
#     test_epoch_loss = 0.0
#     with torch.no_grad():
#         # prev_target = torch.zeros(get_batch_size(interval))
#         for input_sequences, target_value, test_timestamps in test_dataloader:
#             if input_sequences.dim() == 2:
#                 continue
#             model.to(device)
#             # Move data to the appropriate device if using a GPU
#             input_sequences = input_sequences.float().to(device)  # Shape: (batch_size, 20, 3000)
#             # prev_target = prev_target.float().to(device)
#             target_value = target_value.float().to(device)        # Shape: (batch_size,)

#             # Forward pass: Get model predictions
#             output = model(input_sequences).squeeze(1)  # Shape: (batch_size, 1)
#             assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
#             loss = criterion(output, target_value) 
#             test_epoch_loss += loss.item()
#             # prev_target = output.cpu().detach()
#             # Squeeze the output to match target shape
#             if scaler is None:
#                 in_sequence.append(input_sequences.cpu().numpy())
#                 predicted_output.append(output.detach().cpu().numpy())
#                 target_output.append(target_value.cpu().numpy())
#                 timestamps.append(test_timestamps)
#             else:
#                 in_sequence.append(input_sequences.cpu().numpy())
#                 rem = output.cpu().detach().cpu().numpy().shape
#                 predicted_output.append(scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1,1)).reshape(rem))
#                 target_output.append(scaler.inverse_transform(target_value.cpu().numpy().reshape(-1,1)).reshape(rem))
#                 timestamps.append(test_timestamps)
#     print(f"Test loss : {test_epoch_loss / len(test_dataloader)}")
#     end_time = get_current_time()
#     time_to_test = get_time_elapsed(start_time, end_time)
#     print(f"{'End Testing':-^50}")

#     return in_sequence, predicted_output, target_output, timestamps, str(timedelta(seconds=time_to_train))

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, warmup_scheduler, main_scheduler,
                 train_loader, val_loader, test_loader, model_dir,
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
                train_loss, train_mse, train_wmse = self._run_epoch(self.train_loader, training=True)
                val_loss, val_mse, val_wmse = self._run_epoch(self.val_loader, training=False)
            else:
                train_loss = self._run_epoch(self.train_loader, training=True)
                val_loss = self._run_epoch(self.val_loader, training=False)

            if epoch < 5:
                self.warmup_scheduler.step()
                print(f"Epoch {epoch+1} Warmup LR: {self.warmup_scheduler.get_last_lr()[0]:.2e}")
            else:
                self.main_scheduler.step(val_loss)
                print(f"Epoch {epoch+1} Plateau LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if self.monitor1 is not None:
                print(f"\t\tMonitoring -- MSE : {val_mse:.4f}, Weighted MSE : {val_wmse:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                model_path = f"{self.model_dir}/t{self.test_julday}_v{self.val_julday}_{self.interval}_{self.model_type}_model.pt"
                torch.save(best_weights, model_path)
                print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
                consecutive_increase = 0
            else:
                consecutive_increase += 1

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

        for input_sequences, target_value, _ in dataloader:
            if input_sequences.dim() == 2:
                continue

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

            epoch_loss += loss.item()
        if self.monitor1 is not None:
            return epoch_loss / len(dataloader), epoch_mse / len(dataloader), epoch_wmse / len(dataloader)
        else:
            return epoch_loss / len(dataloader)

    def test(self):
        return self._evaluate(self.test_loader, save_path=f"{self.model_dir}/t{self.test_julday}_v{self.val_julday}_{self.interval}_{self.model_type}_model.pt")

    def _evaluate(self, dataloader, save_path):
        self.model.load_state_dict(torch.load(save_path, weights_only=True, map_location=self.device))
        self.model.eval()

        # Print last layer weights
        # if hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Sequential):
        #     print("Last layer weights:")
        #     print(self.model.fc[-1].weight.data)

        in_seq, preds, targets, timestamps = [], [], [], []
        total_loss = 0.0
        if self.monitor1 is not None:
            total_mse = 0.0
            total_wmse = 0.0

        with torch.no_grad():
            for input_sequences, target_value, ts in dataloader:
                if input_sequences.dim() == 2:
                    continue
                input_sequences = input_sequences.float().to(self.device)
                target_value = target_value.float().to(self.device)

                output = self.model(input_sequences).squeeze(1)

                loss = self.criterion(output, target_value)
                total_loss += loss.item()
                if self.monitor1 is not None:
                    total_mse += self.monitor1(output, target_value)
                    total_wmse += self.monitor2(output, target_value)

                pred_unscaled = output.cpu().numpy() * 350
                target_unscaled = target_value.cpu().numpy() * 350

                in_seq.append(input_sequences.cpu().numpy())
                preds.append(pred_unscaled)
                targets.append(target_unscaled)
                timestamps.append(ts)

        print(f"Test Loss: {total_loss / len(dataloader):.4f}")
        if self.monitor1 is not None:
            print(f"\t\tMonitoring -- MSE : {total_mse / len(dataloader):.4f}, Weighted MSE : {total_wmse / len(dataloader):.4f}")
        return in_seq, preds, targets, timestamps, str(timedelta(seconds=self.time_to_train))

