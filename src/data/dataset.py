from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DualViewSample:
    top_path: str
    side_path: str
    weight: float
    cattle_id: str
    split: str


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DualViewWeightDataset(Dataset):
    """
    CSV 格式:
      top_image,side_image,weight,cattle_id,split
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        image_root: Optional[str] = None,
        image_size: int = 640,
        dataframe: Optional[pd.DataFrame] = None,
        train_augment: Optional[bool] = None,
    ):
        if dataframe is not None:
            self.df = dataframe.reset_index(drop=True)
            if train_augment is None:
                train_augment = False
        else:
            self.df = pd.read_csv(csv_path)
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
            if train_augment is None:
                train_augment = split == "train"

        self.root = Path(image_root) if image_root else None
        self.tf = build_transforms(image_size=image_size, train=train_augment)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if self.root is not None and not path.is_absolute():
            path = self.root / path
        return path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        top = Image.open(self._resolve(row["top_image"])).convert("RGB")
        side = Image.open(self._resolve(row["side_image"])).convert("RGB")
        weight = torch.tensor(float(row["weight"]), dtype=torch.float32)
        return {
            "top": self.tf(top),
            "side": self.tf(side),
            "weight": weight,
            "cattle_id": str(row["cattle_id"]),
        }


def split_train_holdout_by_cattle_id(
    csv_path: str,
    holdout_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 split==train 的样本中按 cattle_id 划分出一部分作为验证集，
    避免同一头牛同时出现在训练和验证里。
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1 (exclusive).")

    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No rows with split=='train' in CSV; cannot build hold-out val.")

    rng = np.random.RandomState(seed)
    cattle_ids = train_df["cattle_id"].unique()
    rng.shuffle(cattle_ids)
    n_val = max(1, int(round(len(cattle_ids) * holdout_ratio)))
    val_ids = set(cattle_ids[:n_val])
    mask_val = train_df["cattle_id"].isin(val_ids)
    df_tr = train_df[~mask_val].reset_index(drop=True)
    df_val = train_df[mask_val].reset_index(drop=True)
    if df_tr.empty:
        raise ValueError("Hold-out is too large; training split became empty. Lower holdout_ratio.")
    return df_tr, df_val
