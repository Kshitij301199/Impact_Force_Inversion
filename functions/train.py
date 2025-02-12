import os
import sys
import torch
import torch.optim as optim
import copy

from utils import *

def train_model(model, criterion, optimizer, num_epochs:int, patience:int,
                interval:int, test_julday:int, val_julday:int ,model_type:str,
                train_dataloader, val_dataloader, test_dataloader, model_dir:str, scaler) -> tuple[list, list, list, list]:
    start_time = get_current_time()
    print(f"{'Starting Training':-^50}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = float('inf')
    best_epoch = 0
    consecutive_increase = 0
    best_weights = None
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for input_sequences, target_value, _ in train_dataloader:
            model.to(device)
            input_sequences = input_sequences.float().to(device)  
            target_value = target_value.float().to(device)        
            
            optimizer.zero_grad()
            output = model(input_sequences).squeeze(1)  # Forward pass and shape adjustment
            assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
            loss = criterion(output, target_value)     # Compute loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)  # Average loss for the epoch
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for input_sequences, target_value, _ in val_dataloader:
                model.to(device)
                input_sequences = input_sequences.float().to(device)  
                target_value = target_value.float().to(device)
                # Forward pass: Get model predictions
                output = model(input_sequences).squeeze(1)  
                assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
                loss = criterion(output, target_value) 
                val_epoch_loss += loss.item()
        val_epoch_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val loss : {val_epoch_loss:.4f}, Best Loss : {best_loss:.4f}, RAM usage = {get_memory_usage_in_gb():.2f} GB")
        # Check for early stopping criteria
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())  # Save the best weights
            consecutive_increase = 0  # Reset counter
        else:
            consecutive_increase += 1  # Increment counter

        # Stop training if loss increased for more than early_stop_patience epochs
        if consecutive_increase > patience:
            model.load_state_dict(best_weights)
            print(f"Early stopping triggered. Restoring model to epoch {best_epoch + 1} with lowest loss {best_loss:.4f}")
            break
    
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"{model_dir}/t{test_julday}_v{val_julday}_{interval}_{model_type}_model.pt")
    end_time = get_current_time()
    time_to_train = get_time_elapsed(start_time, end_time)
    print(f"{'End Training':-^50}")

    # Test model
    start_time = get_current_time()
    print(f"{'Start Testing':-^50}")
    model.eval()
    in_sequence, predicted_output, target_output, timestamps = [], [], [], []
    test_epoch_loss = 0.0
    with torch.no_grad():
        for input_sequences, target_value, test_timestamps in test_dataloader:
            model.to(device)
        # Move data to the appropriate device if using a GPU
            input_sequences = input_sequences.float().to(device)  # Shape: (batch_size, 20, 3000)
            target_value = target_value.float().to(device)        # Shape: (batch_size,)

            # Forward pass: Get model predictions
            output = model(input_sequences).squeeze(1)  # Shape: (batch_size, 1)
            assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
            loss = criterion(output, target_value) 
            test_epoch_loss += loss.item()
            # Squeeze the output to match target shape
            if scaler is None:
                in_sequence.append(input_sequences.cpu().numpy())
                predicted_output.append(output.detach().cpu().numpy())
                target_output.append(target_value.cpu().numpy())
                timestamps.append(test_timestamps)
            else:
                in_sequence.append(input_sequences.cpu().numpy())
                rem = output.cpu().detach().cpu().numpy().shape
                predicted_output.append(scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1,1)).reshape(rem))
                target_output.append(scaler.inverse_transform(target_value.cpu().numpy().reshape(-1,1)).reshape(rem))
                timestamps.append(test_timestamps)
    print(f"Test loss : {test_epoch_loss / len(test_dataloader)}")
    end_time = get_current_time()
    time_to_test = get_time_elapsed(start_time, end_time)
    print(f"{'End Testing':-^50}")

    return in_sequence, predicted_output, target_output, timestamps, str(timedelta(seconds=time_to_train))
