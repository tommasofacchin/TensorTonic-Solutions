import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    eps = 1e-12 

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Determine K
    if num_classes is not None:
        K = int(num_classes)
    else:
        if y_true.size == 0 and y_pred.size == 0:
            K = 0
        else:
            max_label = max(int(y_true.max(initial=0)), int(y_pred.max(initial=0)))
            K = max_label + 1

    # Edge case: empty input -> just return zeros
    if y_true.size == 0 or y_pred.size == 0:
        if normalize == 'none':
            return np.zeros((K, K), dtype=int)
        else:
            return np.zeros((K, K), dtype=float)

    # Validate label range [0, K-1]
    if (y_true < 0).any() or (y_true >= K).any():
        raise ValueError("y_true has labels outside [0, K-1]")
    if (y_pred < 0).any() or (y_pred >= K).any():
        raise ValueError("y_pred has labels outside [0, K-1]")

    # Vectorized bincount: index = true*K + pred
    idx = y_true * K + y_pred
    cm = np.bincount(idx, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm.astype(int)

    cm = cm.astype(float)

    if normalize == 'true':
        # Row-wise normalization
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / (denom + eps)

    elif normalize == 'pred':
        # Column-wise normalization
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / (denom + eps)

    elif normalize == 'all':
        denom = cm.sum()
        if denom == 0:
            denom = 1.0
        return cm / (denom + eps)

    else:
        raise ValueError("normalize must be one of {'none','true','pred','all'}")