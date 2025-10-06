"""
Common utilities extracted from EGH444_Assessment2_Template.py
"""

import random

import numpy as np
import torch



def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def colorize_ids(mask: np.ndarray) -> np.ndarray:
    """
    Convert class ID mask to colored visualization.

    Args:
        mask: Class ID mask as numpy array

    Returns:
        RGB colored mask for visualization
    """
    K = int(mask.max()) + 1 if mask.size > 0 else 1
    rng = np.random.RandomState(0)
    cmap = rng.randint(0, 255, (max(K, 1), 3), dtype=np.uint8)
    idx = np.clip(mask, 0, K - 1)
    return cmap[idx]


def to_nyu40_ids(mask_np: np.ndarray) -> np.ndarray:
    """
    Ensure mask uses NYUv2-40 label space [0..39].
    Any pixel outside [0..39] is set to 255 (ignore).

    Args:
        mask_np: Input mask as numpy array

    Returns:
        Mask with values clamped to NYUv2-40 format
    """
    out = mask_np.astype(np.int64, copy=True)
    bad = (out < 0) | (out > 39)
    out[bad] = 255
    return out
