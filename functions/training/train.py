#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-30
# __author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
# __find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission


import sys
import copy
import json
import torch
import numpy as np
from pathlib import Path
from datetime import timedelta

torch.set_default_dtype(torch.float32)

# Dynamic path resolution
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent


def load_config(filename):
    path = project_root / "config" / filename
    with open(path, "r") as file:
        return json.load(file)


try:
    paths = load_config("paths.json")
    data_params = load_config("data_parameters.json")
except FileNotFoundError:
    paths = {"BASE_DIR": str(project_root)}

sys.path.append(paths["BASE_DIR"])
from functions.utils import *


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grad_stats(model):
    """Returns global (mean, max) plus per-submodule norms."""
    all_norms = []
    module_norms = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.norm().item()
        all_norms.append(n)
        # bucket by top-level submodule name, e.g. "embedding", "lstm", "fc_out"
        top = name.split(".")[0]
        module_norms.setdefault(top, []).append(n)

    global_mean = np.mean(all_norms).item() if all_norms else 0.0
    global_max = np.max(all_norms).item() if all_norms else 0.0
    module_max = {k: float(np.max(v)) for k, v in module_norms.items()}
    return global_mean, global_max, module_max


class ModelTrainer:
    """
    A class to handle the training, validation, and testing of a PyTorch model.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        warmup_scheduler,
        main_scheduler,
        train_loader,
        val_loader,
        test_loader,
        model_dir,
        curve_file,
        interval=None,
        test_julday=None,
        val_julday=None,
        model_type="Model",
        device=None,
        warmup_epochs=10,
        grad_clip=1.0,
        grad_instability_threshold=100.0,
    ):

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
        self.time_to_train = 0

        # New: configurable warmup length and grad clipping
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.grad_instability_threshold = grad_instability_threshold

        # New: run-level gradient diagnostics
        self.max_grad_seen = 0.0
        self.min_nonzero_grad_seen = float("inf")
        self.instability_flag = False

        self.model.to(self.device)
        set_seed()

    def train(self, num_epochs, patience):
        best_loss = float("inf")
        best_epoch = 0
        best_weights = None
        consecutive_increase = 0

        start_time = get_current_time()

        for epoch in range(num_epochs):
            train_loss, mean_gs, max_gs, module_gs = self._run_epoch(self.train_loader, training=True, epoch=epoch)
            val_loss, _, _, _ = self._run_epoch(self.val_loader, training=False, epoch=epoch)

            epoch_max_g = max(max_gs) if max_gs else 0.0
            epoch_mean_g = float(np.mean(mean_gs)) if mean_gs else 0.0

            with open(self.curve_file, "a") as file:
                file.write(
                    f"{epoch};{train_loss:.5f};{val_loss:.5f};"
                    f"{self.optimizer.param_groups[0]['lr']:.2e};"
                    f"{epoch_mean_g};{epoch_max_g}\n"
                )

            # Per-batch module-level gradient diagnostics
            if module_gs:
                grad_log_file = self.curve_file.replace(".txt", "_grads.jsonl")
                with open(grad_log_file, "a") as f:
                    f.writelines(
                        json.dumps({"epoch": epoch, "step": step_idx, **mstats}) + "\n"
                        for step_idx, mstats in enumerate(module_gs)
                    )

            # Track instability across the whole run
            self.max_grad_seen = max(self.max_grad_seen, epoch_max_g)
            if epoch_max_g > self.grad_instability_threshold:
                self.instability_flag = True
                print(f"WARNING: grad norm spike {epoch_max_g:.1f} at epoch {epoch + 1}")
            nonzero = [g for g in max_gs if g > 1e-8]
            if nonzero:
                self.min_nonzero_grad_seen = min(self.min_nonzero_grad_seen, min(nonzero))

            # LR schedule: warmup phase (parameterized) then plateau
            if self.warmup_scheduler is None:
                self.main_scheduler.step(val_loss)
                print(f"Epoch {epoch + 1} Plateau LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            elif epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                print(f"Epoch {epoch + 1} Warmup LR: {self.warmup_scheduler.get_last_lr()[0]:.2e}")
            else:
                self.main_scheduler.step(val_loss)
                print(f"Epoch {epoch + 1} Plateau LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if epoch < 50:
                tolerance = 0.05
            elif epoch < 100:
                tolerance = 0.01
            else:
                tolerance = 0.005

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                model_path = (
                    f"{self.model_dir}/{self.model_type}_t{self.test_julday}_v{self.val_julday}_"
                    f"{self.interval}_model.pt"
                )
                torch.save(best_weights, model_path)
                print(f"New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
                consecutive_increase = 0
            elif val_loss <= best_loss + (best_loss * tolerance):
                print(f"Epoch {epoch + 1}: Validation loss {val_loss:.4f} within tolerance of best ({best_loss:.4f}).")
            else:
                consecutive_increase += 1
                print(
                    f"Epoch {epoch + 1}: Validation loss increased ({val_loss:.4f}). Patience {consecutive_increase}/{patience}"
                )

            if consecutive_increase > patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. Best epoch was {best_epoch + 1} with val loss {best_loss:.4f}"
                )
                break

        self.model.load_state_dict(best_weights)
        end_time = get_current_time()
        self.time_to_train = get_time_elapsed(start_time, end_time)

        # Persist a run-level gradient health summary alongside the model
        summary_path = f"{self.model_dir}/{self.model_type}_t{self.test_julday}_v{self.val_julday}_{self.interval}_grad_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"max_grad_seen={self.max_grad_seen:.3f}\n")
            f.write(f"min_nonzero_grad_seen={self.min_nonzero_grad_seen:.3e}\n")
            f.write(f"instability_flag={self.instability_flag}\n")
            f.write(f"best_epoch={best_epoch + 1}\n")
            f.write(f"best_val_loss={best_loss:.5f}\n")

    def _run_epoch(self, dataloader, training=False, epoch=0):
        epoch_loss = 0.0
        self.model.train() if training else self.model.eval()
        mean_gs, max_gs, module_gs = [], [], []
        log_every = 1 if epoch < 5 else 5

        for i, (input_sequences, target_value, _) in enumerate(dataloader):
            if input_sequences.dim() == 2:
                continue
            input_sequences = input_sequences.float().to(self.device)
            target_value = target_value.float().to(self.device)

            if training:
                self.optimizer.zero_grad()

            output = self.model(input_sequences).squeeze(1)
            loss = self.criterion(output, target_value)

            if training:
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                if i % log_every == 0:
                    mean_g, max_g, module_max = grad_stats(self.model)
                    mean_gs.append(mean_g)
                    max_gs.append(max_g)
                    module_gs.append(module_max)

            epoch_loss += loss.item()
        return epoch_loss / len(dataloader), mean_gs, max_gs, module_gs

    def check_train(self, mult_by=50):
        return self._evaluate(
            mult_by,
            self.train_loader,
            save_path=f"{self.model_dir}/{self.model_type}_t{self.test_julday}_v{self.val_julday}_{self.interval}_model.pt",
        )

    def val(self, mult_by=50):
        return self._evaluate(
            mult_by,
            self.val_loader,
            save_path=f"{self.model_dir}/{self.model_type}_t{self.test_julday}_v{self.val_julday}_{self.interval}_model.pt",
        )

    def test(self, mult_by=50):
        return self._evaluate(
            mult_by,
            self.test_loader,
            save_path=f"{self.model_dir}/{self.model_type}_t{self.test_julday}_v{self.val_julday}_{self.interval}_model.pt",
        )

    def _evaluate(self, mult_by, dataloader, save_path):
        self.model.load_state_dict(torch.load(save_path, weights_only=True, map_location=self.device))
        self.model.eval()

        in_seq, preds, targets, timestamps = [], [], [], []
        total_loss = 0.0

        with torch.no_grad():
            for input_sequences, target_value, ts in dataloader:
                if input_sequences.dim() == 2:
                    continue
                input_sequences = input_sequences.float().to(self.device)
                target_value = target_value.float().to(self.device)

                output = self.model(input_sequences).squeeze(1)

                loss = self.criterion(output, target_value)
                total_loss += loss.item()

                pred_unscaled = output.cpu().numpy() * mult_by
                target_unscaled = target_value.cpu().numpy() * mult_by

                in_seq.append(input_sequences.cpu().numpy())
                preds.append(pred_unscaled)
                targets.append(target_unscaled)
                timestamps.append(ts)

        print(f"Test Loss: {total_loss / len(dataloader):.4f}")
        return in_seq, preds, targets, timestamps, str(timedelta(seconds=self.time_to_train))
