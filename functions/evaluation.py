import os
import numpy as np

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Symmetric Mean Absolute Percentage Error (sMAPE)
def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
    Returns:
        float: sMAPE value.
    """
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# Pearson's Correlation Coefficient (PCC)
def pcc(y_true, y_pred):
    """
    Calculate Pearson's Correlation Coefficient (PCC).
    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
    Returns:
        float: PCC value.
    """
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    return numerator / denominator

def evaluate_model(model_type, test_julday, val_julday, interval_seconds, y_true, y_pred, out_dir):
    output_dir = f"{out_dir}/model_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    param1 = mse(y_true, y_pred)
    param2 = smape(y_true, y_pred)
    param4 = pcc(y_true, y_pred)

    with open(f"{output_dir}/evaluation_output.txt", "a") as f:
        f.write(f"{model_type},{test_julday},{val_julday},{interval_seconds},{param1:.4f},{param2:.4f},{param4:.4f}\n")
    return None

