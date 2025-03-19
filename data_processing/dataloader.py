import os
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, input_data, target_data, target_time, interval_count=20, sequence_length=3000):
        """
        Args:
            input_data (np.array or torch.Tensor): Input data of shape (8640000,).
            target_data (np.array or torch.Tensor): Target data of shape (86400,).
            target_time (np.array or torch.Tensor): Timestamps for target data of shape (86400,).
            interval_count (int): Number of intervals per sample (e.g., 20 for 20 intervals).
            sequence_length (int): Length of each interval in input data (3000 for 30 seconds at 100Hz).
        """
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)
        self.target_time = torch.tensor(target_time)
        self.interval_count = interval_count
        self.sequence_length = sequence_length
        self.interval_seconds = int(sequence_length / 100)

    def __len__(self):
        # Each sample corresponds to one target value, so len is the length of target data minus intervals per sample
        return len(self.target_data)

    def __getitem__(self, idx):
        # Get a batch of 20 sequences of 30 seconds each
        sequences = []
        for i in range(self.interval_count):
            start_idx = (idx + i) * 100 * self.interval_seconds
            end_idx = start_idx + self.sequence_length
            sequences.append(self.input_data[start_idx:end_idx])
        # print(start_idx, end_idx)
        # Stack intervals into a (20, 3000) tensor
        input_sequences = torch.stack(sequences, dim=0)  # Shape: (20, 3000)
        target_value = self.target_data[idx]
        target_timestamp = self.target_time[idx]

        return input_sequences, target_value, target_timestamp

# class SequenceDataset(Dataset):
#     def __init__(self, input_data, target_data, target_time, interval_count=20, sequence_length=3000):
#         """
#         Args:
#             input_data (np.array or torch.Tensor): Input data of shape (N,).
#             target_data (np.array or torch.Tensor): Target data of shape (T,).
#             target_time (np.array or torch.Tensor): Timestamps for target data of shape (T,).
#             interval_count (int): Number of intervals per sample.
#             sequence_length (int): Length of each interval in input data.
#         """
#         self.input_data = torch.tensor(input_data, dtype=torch.float32)
#         self.target_data = torch.tensor(target_data, dtype=torch.float32)
#         self.target_time = torch.tensor(target_time)
#         self.interval_count = interval_count
#         self.sequence_length = sequence_length
#         self.interval_seconds = int(sequence_length / 100)  # assuming 100 Hz

#     def __len__(self):
#         return len(self.target_data) - self.interval_count

#     def __getitem__(self, idx):
#         sequences = []
#         prev_targets = []

#         for i in range(self.interval_count):
#             input_idx = idx + i
#             start_idx = input_idx * 100 * self.interval_seconds
#             end_idx = start_idx + self.sequence_length
#             sequences.append(self.input_data[start_idx:end_idx])

#             # Previous target (shifted)
#             prev_idx = input_idx - 1
#             if prev_idx < 0:
#                 prev_targets.append(torch.tensor(0.0))
#             else:
#                 prev_targets.append(self.target_data[prev_idx])

#         input_sequences = torch.stack(sequences, dim=0)  # (20, 3000)
#         prev_targets = torch.tensor(prev_targets).unsqueeze(1)  # (20, 1)
#         target_value = self.target_data[idx]
#         target_timestamp = self.target_time[idx]

#         return input_sequences, prev_targets, target_value, target_timestamp

    
class SequenceDatasetTest(Dataset):
    def __init__(self, input_data, target_time, interval_count=20, sequence_length=3000):
        """
        Args:
            input_data (np.array or torch.Tensor): Input data of shape (8640000,).
            target_time (np.array or torch.Tensor): Timestamps for target data of shape (86400,).
            interval_count (int): Number of intervals per sample (e.g., 20 for 20 intervals).
            sequence_length (int): Length of each interval in input data (3000 for 30 seconds at 100Hz).
        """
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_time = torch.tensor(target_time)
        self.interval_count = interval_count
        self.sequence_length = sequence_length
        self.interval_seconds = int(sequence_length / 100)

    def __len__(self):
        # Each sample corresponds to one target value, so len is the length of target data minus intervals per sample
        return len(self.target_time)

    def __getitem__(self, idx):
        # Get a batch of 20 sequences of 30 seconds each
        sequences = []
        for i in range(self.interval_count):
            start_idx = (idx + i) * 100 * self.interval_seconds
            end_idx = start_idx + self.sequence_length
            sequences.append(self.input_data[start_idx:end_idx])
        # print(start_idx, end_idx)
        # Stack intervals into a (20, 3000) tensor
        input_sequences = torch.stack(sequences, dim=0)  # Shape: (20, 3000)
        target_timestamp = self.target_time[idx]

        return input_sequences, target_timestamp