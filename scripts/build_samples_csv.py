"""
根据固定目录结构 + 体重表自动生成 data/samples.csv

目录约定（在 --photos-root 下面）::
    train/<cattle_id>/top.jpg   （或 .png）
    train/<cattle_id>/side.jpg
    test/<cattle_id>/top.jpg
    test/<cattle_id>/side.jpg

体重表 CSV（--weights）：两列 cattle_id,weight（公斤）

生成的 paths 相对于 --manifest-root（与 configs/train.yaml 里 data.image_root 一致时用 data）

用法示例::

    python scripts/build_samples_csv.py ^
      --photos-root data/images ^
      --manifest-root data ^
      --weights data/weights.csv ^
      --out data/samples.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _find_named_image(folder: Path, name_prefix: str) -> Path:
    matches: List[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = folder / f"{name_prefix}{ext}"
        if p.is_file():
            matches.append(p)
    if len(matches) == 1:
        return matches[0]
    # 兼容 TOP.JPG 等大小写（Windows）
    for p in folder.iterdir():
        if p.is_file() and p.stem.lower() == name_prefix.lower():
            return p
    raise FileNotFoundError(f'在 {folder} 中找不到 {name_prefix}.(jpg/jpeg/png/webp/bmp)')


def collect_pairs(photos_root: Path) -> List[Tuple[str, str, Path, Path]]:
    rows: List[Tuple[str, str, Path, Path]] = []
    for split in ("train", "test"):
        sd = photos_root / split
        if not sd.is_dir():
            continue
        for cattle_dir in sorted(sd.iterdir()):
            if not cattle_dir.is_dir():
                continue
            cattle_id = cattle_dir.name
            top_p = _find_named_image(cattle_dir, "top")
            side_p = _find_named_image(cattle_dir, "side")
            rows.append((split, cattle_id, top_p, side_p))
    if not rows:
        raise RuntimeError(
            f"未在 {photos_root} 下找到 train/*/top.* 或 test/*/top.*；"
            "请按脚本顶部说明摆放图片。"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="从文件夹 + 体重表生成 samples.csv")
    parser.add_argument("--photos-root", type=str, default="data/images", help="含 train/test 子目录")
    parser.add_argument(
        "--manifest-root",
        type=str,
        default="data",
        help="manifest 里写的相对路径根目录（应对应 train.yaml 的 image_root）",
    )
    parser.add_argument("--weights", type=str, required=True, help="CSV：cattle_id,weight")
    parser.add_argument("--out", type=str, default="data/samples.csv")
    args = parser.parse_args()

    photos_root = Path(args.photos_root)
    manifest_root = Path(args.manifest_root).resolve()
    weights = pd.read_csv(args.weights)
    if list(weights.columns[:2]) != ["cattle_id", "weight"]:
        weights = weights.rename(
            columns={weights.columns[0]: "cattle_id", weights.columns[1]: "weight"}
        )
    weight_map = {
        str(r["cattle_id"]): float(r["weight"]) for _, r in weights.iterrows()
    }

    records = []
    for split, cattle_id, top_abs, side_abs in collect_pairs(photos_root.resolve()):
        if cattle_id not in weight_map:
            raise KeyError(f"weights.csv 缺少 cattle_id={cattle_id!r}")
        top_rel = top_abs.resolve().relative_to(manifest_root).as_posix()
        side_rel = side_abs.resolve().relative_to(manifest_root).as_posix()
        records.append(
            {
                "top_image": top_rel,
                "side_image": side_rel,
                "weight": weight_map[cattle_id],
                "cattle_id": cattle_id,
                "split": split,
            }
        )

    out_df = pd.DataFrame.from_records(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
