import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DualViewWeightDataset, split_train_holdout_by_cattle_id
from models.two_stream import TwoStreamCBAMResNet50SE
from utils.metrics import regression_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        top = batch["top"].to(device)
        side = batch["side"].to(device)
        y = batch["weight"].to(device)
        pred = model(top, side)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return regression_metrics(y_true, y_pred)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    val_cfg = cfg.get("validation") or {}
    mode = val_cfg.get("mode", "val")

    csv_path = cfg["data"]["csv_path"]
    image_root = cfg["data"].get("image_root")
    image_size = cfg["data"]["image_size"]

    if mode == "val":
        train_ds = DualViewWeightDataset(
            csv_path=csv_path,
            split="train",
            image_root=image_root,
            image_size=image_size,
        )
        val_ds = DualViewWeightDataset(
            csv_path=csv_path,
            split="val",
            image_root=image_root,
            image_size=image_size,
        )
    elif mode == "test":
        train_ds = DualViewWeightDataset(
            csv_path=csv_path,
            split="train",
            image_root=image_root,
            image_size=image_size,
        )
        val_ds = DualViewWeightDataset(
            csv_path=csv_path,
            split="test",
            image_root=image_root,
            image_size=image_size,
        )
    elif mode == "train_holdout":
        df_tr, df_val = split_train_holdout_by_cattle_id(
            csv_path,
            holdout_ratio=float(val_cfg.get("holdout_ratio", 0.15)),
            seed=cfg["seed"],
        )
        train_ds = DualViewWeightDataset(
            csv_path,
            split="train",
            image_root=image_root,
            image_size=image_size,
            dataframe=df_tr,
            train_augment=True,
        )
        val_ds = DualViewWeightDataset(
            csv_path,
            split="train",
            image_root=image_root,
            image_size=image_size,
            dataframe=df_val,
            train_augment=False,
        )
    elif mode == "none":
        train_ds = DualViewWeightDataset(
            csv_path=csv_path,
            split="train",
            image_root=image_root,
            image_size=image_size,
        )
        val_ds = None
    else:
        raise ValueError(
            "validation.mode must be one of: val, test, train_holdout, none"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )

    model = TwoStreamCBAMResNet50SE(
        pretrained_backbone=cfg["model"]["pretrained_backbone"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.L1Loss() if cfg["train"]["loss"] == "mae" else nn.MSELoss()
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    metric_key = "rmse"
    best_metric = float("inf")
    best_train_loss = float("inf")
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        running_loss = 0.0
        for batch in pbar:
            top = batch["top"].to(device, non_blocking=True)
            side = batch["side"].to(device, non_blocking=True)
            y = batch["weight"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                pred = model(top, side)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * top.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_ds)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            if val_metrics[metric_key] < best_metric:
                best_metric = val_metrics[metric_key]
                torch.save(
                    {"model": model.state_dict(), "config": cfg},
                    out_dir / "best.pt",
                )
        else:
            record = {"epoch": epoch, "train_loss": train_loss}
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(
                    {"model": model.state_dict(), "config": cfg},
                    out_dir / "best.pt",
                )

        history.append(record)
        print(record)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    if val_loader is not None:
        print(f"Training finished. Best {metric_key.upper()}: {best_metric:.4f}")
    else:
        print(f"Training finished (no val). Best train loss: {best_train_loss:.4f}")


if __name__ == "__main__":
    main()
