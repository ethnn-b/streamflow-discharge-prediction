import numpy as np

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics (RMSE, MAE, R2, NSE, KGE)."""
    # Filter out NaN values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    ss_total = np.sum((y_true_clean - np.mean(y_true_clean))**2)
    ss_residual = np.sum((y_true_clean - y_pred_clean)**2)
    
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan
    nse = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

    # Calculate Kling Gupta Efficiency (KGE)
    kge = calculate_kge(y_true_clean, y_pred_clean)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': nse, 'KGE': kge}

def calculate_kge(y_true, y_pred):
    """Calculate Kling Gupta Efficiency (KGE).
    
    KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)
    
    Where:
    - r = correlation coefficient
    - beta = mean(y_pred) / mean(y_true) (bias ratio)
    - gamma = std(y_pred) / std(y_true) (variability ratio)
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Correlation coefficient
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        r = 0
    
    # Bias ratio (beta)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    if mean_true == 0:
        beta = np.nan if mean_pred != 0 else 1.0
    else:
        beta = mean_pred / mean_true
    
    # Variability ratio (gamma)
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true == 0:
        gamma = np.nan if std_pred != 0 else 1.0
    else:
        gamma = std_pred / std_true
    
    # Handle NaN values in beta or gamma
    if np.isnan(beta) or np.isnan(gamma):
        return np.nan
    
    # Calculate KGE
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    return kge
