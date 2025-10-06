"""
NYUv2 Depth Dataset module extracted from EGH444_Assessment2_Template.py
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np
import cv2
from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):
    """
    NYUv2-style depth dataset for semantic segmentation.

    Expects directory structure:
      base/
        RGB/{stem}.jpg|png
        Depth/{stem}.png
        Label/{stem}.png      # semantic class IDs
        train.txt / test.txt  # lines with filenames or paths; we parse .stem

    Returns raw arrays for transformation by the transform function.
    """

    def __init__(
        self, base_dir: str, split: str = "train", transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize NYUDepthDataset.

        Args:
            base_dir: Path to dataset root directory
            split: Dataset split ('train' or 'test')
            transform: Optional transform function to apply to samples
        """
        self.base = Path(base_dir)
        self.rgb_dir = self.base / "RGB"
        self.dep_dir = self.base / "Depth"
        self.lbl_dir = self.base / "Label"
        self.transform = transform

        # Read split file and extract stems
        with open(self.base / f"{split}.txt") as f:
            stems = [Path(line.split()[0]).stem for line in f if line.strip()]

        # Find valid samples (all three files must exist)
        self.items = []
        for s in stems:
            rp = self.rgb_dir / f"{s}.jpg"
            if not rp.exists():
                rp = self.rgb_dir / f"{s}.png"
            dp = self.dep_dir / f"{s}.png"
            lp = self.lbl_dir / f"{s}.png"
            if rp.exists() and dp.exists() and lp.exists():
                self.items.append((s, rp, dp, lp))

        if not self.items:
            raise RuntimeError("No valid samples found. Check paths / split file.")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - rgb: RGB image as numpy array (H, W, 3)
                - depth: Depth image as numpy array (H, W)
                - mask: Label mask as numpy array (H, W)
                - id: Sample identifier string
        """
        stem, rp, dp, lp = self.items[idx]

        # Load RGB image and convert from BGR to RGB
        rgb = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth as float32
        depth = cv2.imread(str(dp), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Load label mask as int64
        label = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE).astype(np.int64)

        sample = {"rgb": rgb, "depth": depth, "mask": label, "id": stem}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
