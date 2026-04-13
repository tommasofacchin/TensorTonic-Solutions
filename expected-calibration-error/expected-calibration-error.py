import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error (ECE) with equal-width bins on [0, 1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-15
    y_pred = np.minimum(y_pred, 1.0 - eps)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[:-1], right=False) - 1  

    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue 

        bin_true = y_true[mask]
        bin_pred = y_pred[mask]

        acc = bin_true.mean()
        conf = bin_pred.mean()
        weight = len(bin_true) / n

        ece += weight * abs(acc - conf)

    return ece