import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import DualViewWeightDataset
from models.two_stream import TwoStreamCBAMResNet50SE
from utils.metrics import regression_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--ckpt", type=str, default="runs/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    model = TwoStreamCBAMResNet50SE(
        pretrained_backbone=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = DualViewWeightDataset(
        csv_path=cfg["data"]["csv_path"],
        split=args.split,
        image_root=cfg["data"].get("image_root"),
        image_size=cfg["data"]["image_size"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    ys, ps = [], []
    for batch in loader:
        top = batch["top"].to(device)
        side = batch["side"].to(device)
        y = batch["weight"].to(device)
        pred = model(top, side)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    metrics = regression_metrics(np.concatenate(ys), np.concatenate(ps))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
