"""
utility functions come here
"""

import numpy as np

def int_range(a:int, b:int) -> np.ndarray:
    return np.linspace(a, b, b-a+1, dtype=int)