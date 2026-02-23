import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    w = np.zeros((n_features,))
    b = 0
    for _ in range(steps):
        z =  np.matmul(X, w) + b
        p = _sigmoid(z)
        w = w - lr * 1/n_samples * np.matmul(X.T, p - y)
        b = b - lr * np.mean(p - y)       
    return w, b