# 基于双视角 RGB 图像的肉牛体重估测

（英文版说明见 **`README.md`**）

本仓库为肉牛双视角 RGB 图像体重估测工作的参考实现：实例分割模型 **EMA-YOLO11n-seg**（需使用 Ultralytics 等框架另行训练），以及 **双流 CBAM-ResNet50-SE** 回归网络。

**目标。** 给定同一头牛、同一成像时刻对齐的顶视图（`top`）与侧视图（`side`）RGB 图像对，输出**活重（kg）**。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `configs/train.yaml` | 训练超参、`image_root` 与验证策略 |
| `data/samples.csv` | 配对清单示例（`top_image`, `side_image`, `weight`, `cattle_id`, `split`） |
| `scripts/build_samples_csv.py` | 由目录结构 + `weights.csv` 生成 `samples.csv` |
| `src/models/attention.py` | CBAM 与轻量 SE 融合 |
| `src/models/two_stream.py` | 双流主干 + 融合 + 回归头 |
| `src/train.py` | 训练脚本（支持混合精度 AMP） |
| `src/evaluate.py` | 在指定划分上评估（`train` / `val` / `test`） |
| `src/infer.py` | 单对图像推理 |

## 环境

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## `samples.csv` 格式

每一**行**表示一个**配对样本**：同一头牛、**同一采样时刻**，顶视（背面轮廓）与侧视（腋腹形态）两张图。

| 列名 | 说明 |
| --- | --- |
| `top_image` | 顶视图相对路径（相对 `configs/train.yaml` 中 `data.image_root`，通常为 `data`） |
| `side_image` | 侧视图相对路径 |
| `weight` | 地磅等场所测得的体重真值（**kg**） |
| `cattle_id` | 牛只编号。划分 train/test 时 **按编号互斥**，避免同一头牛泄露 |
| `split` | `train`，可选 `val`，或 `test` |

**示例。** 磁盘路径为 `<仓库>/code/data/images/train/cow_001/top.jpg`，且配置中 `image_root: data`，则 CSV 中写：

`images/train/cow_001/top.jpg` 与 `images/train/cow_001/side.jpg`。

## 如何准备配对样本

### 方式 A：表格 / CSV（最灵活）

每一对图一行，字段见上表。确保同一 `cattle_id` 不要同时出现在 `train` 与 `test`。

### 方式 B：文件夹 + 自动生成清单

1. 按如下目录组织图像：

```text
data/images/
  train/<cattle_id>/top.jpg   （亦支持 .png / .jpeg 等）
        /<cattle_id>/side.jpg
  test/<cattle_id>/top.jpg
       /<cattle_id>/side.jpg
```

2. 准备 `data/weights.csv`，列为 `cattle_id,weight`，可参考 `data/weights_TEMPLATE.csv`。

3. 在仓库根目录执行：

```bash
python scripts/build_samples_csv.py \
  --photos-root data/images \
  --manifest-root data \
  --weights data/weights.csv \
  --out data/samples.csv
```

（Windows CMD 可将行末 `\` 换为 `^` 续行。）

4. 确认 `configs/train.yaml` 中 `data.image_root` 与 `--manifest-root` 一致（默认均为 `data`）。

**局限。** 辅助脚本默认在每个 `split` 下的每头牛只对应 **一对** `top/side`。若同一牛只多个时刻有多对图，可扩展子目录逻辑，或直接手写多行 CSV。

## 仅有 `train` / `test`、无独立验证集时

在 `configs/train.yaml` 中配置 **`validation`**：

| `validation.mode` | 行为 |
| --- | --- |
| **`train_holdout`（默认）** | 在 `split==train` 中按 **`cattle_id`** 随机留出若干个体作为验证集（比例 `holdout_ratio`）。**测试集不参与选模型**，推荐写法。 |
| `test` | 每轮在 **test** 上计算指标并保存最优权重，相当于用测试集参与模型选择；除非有特殊说明，一般**不推荐**用于无偏报告。 |
| `none` | 不做验证；仅按 **训练损失** 保存 `best.pt`。 |
| `val` | 当 CSV 中存在 `split==val` 时使用。 |

默认片段：

```yaml
validation:
  mode: train_holdout
  holdout_ratio: 0.15
```

## 训练

```bash
python src/train.py --config configs/train.yaml
```

输出：

- `runs/best.pt`：校验可用时为验证 RMSE 最优；否则为训练损失最优
- `runs/history.json`：逐轮记录

## 评估（例如在独立测试牛只上）

```bash
python src/evaluate.py --config configs/train.yaml --ckpt runs/best.pt --split test
```

控制台打印 JSON，含 **MAE**、**RMSE**、**R²**、**MAPE**（见 `src/utils/metrics.py`）。

## 推理（单对图像）

```bash
python src/infer.py --config configs/train.yaml --ckpt runs/best.pt \
  --top path/to/top.jpg --side path/to/side.jpg
```

## 与论文方法的对应关系

- **双流融合 + CBAM + SE 式通道重标定：** `src/models/two_stream.py`、`src/models/attention.py`
- **分割阶段（EMA-YOLO11n-seg）：** 在 Ultralytics 上训练实例分割 → 导出去背景或非背景干扰的裁剪图 → 将路径填入 `samples.csv`
- **评价指标：** `src/utils/metrics.py`

## 引用

若在论文中使用本代码，请引用对应正式发表稿件；并酌情引用 PyTorch、torchvision、Ultralytics YOLO 等第三方库。

## 版本管理

```bash
git init
git add .
git commit -m "Dual-view beef cattle weight estimation code release"
```

大体积文件（`runs/`、`*.pt`、原始 `data/images/` 等）已列入 `.gitignore`；权重与完整数据集可另行托管。

## 许可证

公开发布前请按单位与编辑部要求填写合适的开源许可证及数据使用协议。
