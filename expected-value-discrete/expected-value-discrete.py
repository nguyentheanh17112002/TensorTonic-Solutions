import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    tolerance = 1e-6
    if np.sum(p) < 1 - tolerance or np.sum(p) > 1 + tolerance:
        raise ValueError
    res = np.dot(x, p)
    return res 