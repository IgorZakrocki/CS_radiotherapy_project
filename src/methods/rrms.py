import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error (MSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)

def rrms(y_true, y_pred, eps: float = 1e-12):
    """Relative Root Mean Square error (RRMS)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.sqrt(np.mean(y_true ** 2)) + eps
    return num / den
