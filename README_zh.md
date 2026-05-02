# 基于双视角 RGB 图像的肉牛活重估测

**英文对照首页见：** `README.md`

---

## 项目简介

- **要解决什么：** 在固定双相机场景下，用同一时刻对齐的「顶视图 + 侧视图」RGB 图像，回归预测肉牛**活重（kg）**，减少对地磅驱赶与应激。

- **方法梗概：**（1）**实例分割**（如 **EMA-YOLO11n-seg**，建议用 **Ultralytics** 单独训练），减弱栏舍与地面等背景干扰；（2）**双流 CBAM-ResNet50-SE** —— 分别从顶视、侧视提特征并在融合后对通道加权，经全连接回归得到体重。

- **数据拆分：** **`samples.csv` 里每条记录对应同一时间戳的一对图像 + 标签**；务必按 **`cattle_id` 互斥划分 train/test，避免个体差异泄露。** 若没有单独 **val**，在 `configs/train.yaml` 中默认 **`validation.mode: train_holdout`**（从训练集中的牛号随机留出验证）。

---

## 仓库里有什么（文件说明）

根目录：`requirements.txt`、`README.md`（中英首页）、本文 `README_zh.md`、`.gitignore`。

### 目录

- **`configs/train.yaml`**：训练(epochs、lr、batch、AMP)、数据根目录 `image_root`、验证策略 `validation`。

- **`data/`**：示例清单 `samples.csv`、`weights_TEMPLATE.csv`（复制改名为 `weights.csv`），以及 `images/` 下放原始或分割后图像的子目录占位说明。

- **`scripts/build_samples_csv.py`**：给定 `train|test/<cattle_id>/top.*` `side.*` 与 **`weights.csv`**，自动生成 **`data/samples.csv`**。

- **`src/`**：`models/`、`data/`、`utils/` 与 **`train.py` / `evaluate.py` / `infer.py`**。

### 源码要点

| 路径 | 作用 |
|------|------|
| `src/models/two_stream.py` | 双流 CBAM–ResNet + 融合 SE + 回归头 |
| `src/models/attention.py` | CBAM、轻量 SE |
| `src/data/dataset.py` | Dataset；`split_train_holdout_by_cattle_id` |
| `src/utils/metrics.py` | MAE、RMSE、R²、MAPE |
| `src/train.py` | 训练，输出 `runs/best.pt`、`history.json` |

---

## 快速命令

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows
pip install -r requirements.txt

python src/train.py    --config configs/train.yaml
python src/evaluate.py --config configs/train.yaml --ckpt runs/best.pt --split test
python src/infer.py    --config configs/train.yaml --ckpt runs/best.pt --top 顶视图.jpg --side 侧视图.jpg
```

CSV 字段：`top_image`、`side_image`、`weight`、`cattle_id`、`split`。路径相对配置文件中的 `data.image_root`（通常为 **`data`**）。目录自动生成示例见 **`scripts/build_samples_csv.py`**。

---

## 验证模式（摘自 `configs/train.yaml`）

| `validation.mode` | 说明 |
|-------------------|------|
| `train_holdout` | 默认。从 **`split==train`** 牛号里按比例留出验证，**不参与 test**。**推荐。** |
| `test`          | 用测试集当作验证选模型——一般用于内部实验，不推荐作为正式报告惯例。 |
| `none`          | 仅用训练损失选 checkpoint。 |
| `val`           | CSV 中提供 `split=val` 时使用。 |

---

## 投稿与开源

文稿录用后请将 **稿件题目、作者顺序、年份 DOI** 补进 `README.md` 与本文的引用段落；数据集若涉隐私，请勿将原始图像或未脱敏 CSV 入库，可仅存示例或链接。
