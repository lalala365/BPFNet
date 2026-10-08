# BPFNet: Reciprocal Perception Bidirectional Fusion with Polar-Frequency Structural Enhancement for Referring Camouflaged Object Detection

> PyTorch implementation of BPFNet for referring camouflaged object detection (Ref-COD). This repository provides the BPFNet model implementation and evaluation script. The architecture below follows the **updated Figure 2**, which takes precedence over the earlier illustration in the manuscript.

## Overview

Referring Camouflaged Object Detection (Ref-COD) segments the camouflaged object specified by a reference image. BPFNet combines reference-guided cross-image interaction, polar-frequency structural enhancement, and multi-scale prediction. The implementation uses a PVTv2-B2 image backbone and reference features supplied by a separate, precomputed reference-feature pipeline.

## Architecture

![BPFNet architecture](BPFNet/architecture.png)

*Figure 2. Updated BPFNet architecture (author-provided).* 

The updated architecture contains:

- **PVTv2-B2 backbone:** extracts four stages of features from the camouflaged image.
- **Reciprocal Perception Bidirectional Fusion (RPBF):** uses bidirectional token-level cross-attention and two independently learned sigmoid gates to integrate camouflaged-image and reference features. In the supplied implementation, this is `SoftGateCrossAttention` (with `TokenCrossAttention`).
- **Polar-Frequency Structural Enhancement (PFSE):** enhances fused features through radial-angular frequency attention. The implementation uses `PolarSectorSelfAttn` (RAFA) with configurable stage-wise, weak residual injection. By default, enhancement is enabled at all four stages (`0123`).
- **Transformer encoder and lightweight decoder:** process the multi-stage representations and generate four segmentation predictions. The decoder implementation reuses a lightweight transformer decoder across the stages.
- **Error-Guided Uncertainty Calibration Branch (EUCB):** during training, predicts an uncertainty map and supervises it using the absolute error between the segmentation probability and ground truth. This auxiliary branch is not required for test-time output.

## Repository Layout

```text
BPFNet/
├── README.md
├── assets/
│   └── architecture.png
├── models/
│   ├── __init__.py
│   ├── UAT.py                 # BPFNet.py renamed to match the supplied test import
│   └── pvtv2.py
├── test.py                   # renamed from test (1).py
├── data.py                   # REQUIRED: dataset loader (not yet provided)
├── pvt_weights/
│   └── pvt_v2_b2.pth         # pretrained backbone weights (not yet provided)
└── checkpoints/
    └── best.pth              # best model checkpoint (download separately)
```

**Important:** `data.py`, the reference-feature extraction/preprocessing pipeline, and the pretrained backbone checkpoint are external dependencies of the supplied code and are not bundled with this starter package. The author has confirmed they are available in the working environment; they are still needed by anyone running this repository from scratch. The `models/UAT.py` path above deliberately matches the existing `from models.UAT import Network` in `test.py`; alternatively, change the import to match your chosen filename.

## Requirements

The supplied code imports:

- Python 3
- PyTorch (`torch`)
- `timm` (PVTv2/DropPath utilities)
- `numpy`
- `tqdm`
- `matplotlib`
- `py-sod-metrics` (`py_sod_metrics`)

Install the main dependencies with:

```bash
pip install torch torchvision timm numpy tqdm matplotlib py-sod-metrics
```

Exact versions and CUDA configuration have not been verified against the original training environment. The provided PVTv2 code uses legacy `timm.models.*` imports, so a compatible `timm` version may be required.

## Pretrained Weights

### BPFNet checkpoint

The best trained BPFNet checkpoint is **`best.pth`** (approximately 707 MiB). Because of its file size, the checkpoint is hosted on Baidu Netdisk rather than included directly in this GitHub repository.

| Checkpoint | Dataset | Download | Extraction code |
| --- | --- | --- | --- |
| `best.pth` | R2C7K | [Baidu Netdisk](https://pan.baidu.com/s/1qgkSpdOZvWVG7YuE5TgXVw?pwd=asfj) | `asfj` |

**Download steps:**

1. Open the Baidu Netdisk link and enter the extraction code `asfj` if requested.
2. Download `best.pth` and place it at `checkpoints/best.pth` under the repository root.
3. Keep the filename `best.pth`, or adjust the `--ckpt` argument to match its location.

> The shared checkpoint is provided for evaluation/inference. The Baidu Netdisk link and extraction code were supplied by the project author; remote download accessibility has not been independently verified.

### PVTv2-B2 backbone

The model constructor also loads pretrained backbone weights from `./pvt_weights/pvt_v2_b2.pth` **before** loading the BPFNet checkpoint. Place a compatible PVTv2-B2 checkpoint there or set `--pvt_weights` to its path. A download source for the exact backbone checkpoint has not yet been supplied.

## Dataset

The manuscript evaluates BPFNet on **R2C7K**. The supplied test script expects a `get_dataloader` function in `data.py`, returning batches of:

```python
images, gts, ref_feats, metadata
```

`ref_feats` must be a tuple/list of four tensors corresponding to the four reference feature stages. According to the updated Figure 2, reference features originate from a pretrained ICON reference branch; the provided three Python files do not contain its full feature-generation pipeline.

Set `--data_root` to your local dataset location. Dataset download and preprocessing instructions will be added when the loader and reference feature preparation code are available.

## Evaluation

With the project dependencies, dataset loader, reference features and pretrained PVTv2-B2 backbone weights prepared, evaluate the downloaded `best.pth` using:

```bash
python test.py \
  --data_root /path/to/R2C7K \
  --ckpt ./checkpoints/best.pth \
  --pvt_weights ./pvt_weights/pvt_v2_b2.pth \
  --imgsize 352 \
  --shot 5 \
  --batchsize 1 \
  --dim 64 \
  --use_polar 1 \
  --rafa_stages 0123 \
  --rafa_max_beta 0.01 \
  --rafa_beta_init 0.001 \
  --rafa_gamma_init 0.1 \
  --out_dir ./eval_results
```

The evaluation script supports a single checkpoint via `--ckpt`, or a list of epoch checkpoints via `--save_root` and `--epochs`. It selects the final segmentation output `s0`, resizes it to the ground-truth size, applies sigmoid, and calculates the metrics below.

### Evaluation metrics

- S-measure (`Smeasure`)
- Weighted F-measure (`WeightedFmeasure`)
- Mean absolute error (`MAE`)
- Adaptive / mean / maximum E-measure
- Adaptive / mean / maximum F-measure
- A supplementary selection score: `Smeasure + WeightedFmeasure - MAE`

The script writes text/CSV evaluation results and a performance curve to the selected output directory. **Numerical benchmark results are not listed here because no verified evaluation log was supplied.**

## Implementation Notes

- The supplied network uses four separate fusion modules and four separate RAFA/PFSE instances.
- RAFA uses polar sectors (default: 8 radial bins and 16 angular bins), a transformer encoder, and a conservative learnable residual contribution.
- `--rafa_stages` controls the stages where PFSE is applied; `0123` enables all four.
- During training, the uncertainty head is supervised by an L1 loss against a detached pixel-wise prediction-error map. The full training script and full segmentation-loss implementation were not provided.
- The supplied checkpoint loader uses `strict=False` and includes compatibility handling for older RAFA key names. Always inspect reported missing/unexpected keys to confirm the intended weights loaded.

## Paper

**A Reciprocal Perception Bidirectional Fusion with Polar-Frequency Structural Enhancement for Referring Camouflaged Object Detection**

Manuscript provided by the authors. Please update the publication metadata and citation after acceptance/publication.

## Citation

Citation details will be added when a final bibliographic record is available.

## Acknowledgements

This implementation uses PVTv2-B2 as its image backbone and reference features from an ICON-based pipeline. Please also acknowledge the original backbone, reference-feature method, and dataset in the final public release.
