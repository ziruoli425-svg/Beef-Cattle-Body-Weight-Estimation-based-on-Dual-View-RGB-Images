import argparse

import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.two_stream import TwoStreamCBAMResNet50SE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--ckpt", type=str, default="runs/best.pt")
    parser.add_argument("--top", type=str, required=True, help="Top-view image path")
    parser.add_argument("--side", type=str, required=True, help="Side-view image path")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamCBAMResNet50SE(pretrained_backbone=False, dropout=cfg["model"]["dropout"]).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    top = tf(Image.open(args.top).convert("RGB")).unsqueeze(0).to(device)
    side = tf(Image.open(args.side).convert("RGB")).unsqueeze(0).to(device)

    pred = model(top, side).item()
    print(f"Predicted body weight: {pred:.2f} kg")


if __name__ == "__main__":
    main()
