# Please do not distribute this code without the author's permission

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset

def concat_sequence_datasets(datasets):
    """
    Combine multiple SequenceDataset / SequenceDatasetTest instances into
    a single dataset that can be passed to one DataLoader.

    Args:
        datasets (list[Dataset]): list of dataset instances, e.g. one
            SequenceDataset per event/julday, all built with the same
            interval_count / sequence_length / sampling_rate.

    Returns:
        ConcatDataset: a dataset whose __len__ is the sum of the inputs'
        lengths, and whose __getitem__ transparently dispatches to the
        correct underlying dataset.
    """
    if len(datasets) == 0:
        raise ValueError("concat_sequence_datasets received an empty list.")
    return ConcatDataset(datasets)

class BaseSequenceDataset(Dataset):
    """Base class for sequence datasets to handle shared logic efficiently."""
    def __init__(self, input_data, target_time, interval_count=20, sequence_length=3000, sampling_rate=100):
        """
        Args:
            input_data (np.ndarray or torch.Tensor): 1D input seismic data.
            target_time (np.ndarray or torch.Tensor): Timestamps for each window.
            interval_count (int): Number of intervals per sample (e.g., 20 for 20 intervals).
            sequence_length (int): Samples per interval (e.g., 3000 for 30s @ 100Hz).
            sampling_rate (int): Sampling rate of input data in Hz.
        """
        # Ensure numpy inputs are writable / own their memory to avoid PyTorch warning
        if isinstance(input_data, np.ndarray):
            input_data = np.asarray(input_data).astype(np.float32, copy=False)
            # make a writable copy if the array is not writable
            if not input_data.flags.writeable:
                input_data = input_data.copy()
        self.input_data = torch.as_tensor(input_data, dtype=torch.float32)

        if isinstance(target_time, np.ndarray):
            target_time = np.array(target_time, copy=False)
            if not target_time.flags.writeable:
                target_time = target_time.copy()
        self.target_time = torch.as_tensor(target_time)
        self.interval_count = interval_count
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate

    def __len__(self):
        # Calculate how many full sliding windows of 'interval_count' we can fit
        max_chunks = len(self.input_data) // self.sequence_length
        available_windows = max_chunks - self.interval_count + 1
        return max(0, min(len(self.target_time), available_windows))

    def _get_input_sequence(self, idx):
        # Efficient vectorized slicing: slice once, then reshape
        start_idx = idx * self.sequence_length
        total_samples = self.interval_count * self.sequence_length
        end_idx = start_idx + total_samples
        
        chunk = self.input_data[start_idx:end_idx]
        # ensure contiguous memory before view
        chunk = chunk.contiguous()
        # .view() is zero-copy compared to looping with torch.stack()
        return chunk.view(self.interval_count, self.sequence_length)

class SequenceDataset(BaseSequenceDataset):
    def __init__(self, input_data, target_data, target_time, interval_count=20, sequence_length=3000, sampling_rate=100):
        super().__init__(input_data, target_time, interval_count, sequence_length, sampling_rate)
        self.target_data = torch.as_tensor(target_data, dtype=torch.float32)

    def __getitem__(self, idx):
        input_sequences = self._get_input_sequence(idx)
        target_value = self.target_data[idx]
        target_timestamp = self.target_time[idx]
        return input_sequences, target_value, target_timestamp

class SequenceDatasetTest(BaseSequenceDataset):
    def __init__(self, input_data, target_time, interval_count=20, sequence_length=3000, sampling_rate=100):
        super().__init__(input_data, target_time, interval_count, sequence_length, sampling_rate)

    def __len__(self):
        # Test dataset length usually depends strictly on provided timestamps
        return len(self.target_time)

    def __getitem__(self, idx):
        input_sequences = self._get_input_sequence(idx)
        target_timestamp = self.target_time[idx]
        return input_sequences, target_timestamp
