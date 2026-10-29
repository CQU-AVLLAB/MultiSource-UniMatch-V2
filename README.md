# Multi-Source Uncertainty-Guided Semi-Supervised Segmentation

Research implementation built on UniMatch V2 (Lihe Yang et al.), commit c05643fa559bae5996b2d07d6345f47667bf4405. The upstream MIT license and attribution are retained. See README_UPSTREAM.md for upstream documentation and citation.

## Status
Work in progress. Full 40,000-step training is not complete. No claim of reproducing final paper scores is made. This repository distributes source only: no weights, datasets, training outputs, or credentials.

## Method
EMA teacher normalized predictive entropy and normalized Jensen-Shannon weather divergence are fused to form uncertainty. Confidence is penalized exponentially to select pseudo-labels, with continuous uncertainty loss weights and student weather-consistency regularization. See util/multi_source.py for exact perturbation parameters. Fog uses a synthetic smooth random 4x4 proxy map, not measured/predicted depth. At batch size one, batch-reversal CutMix is self-mixing and does not mix distinct images.

## Setup
Tested with Python 3.10, PyTorch 2.4.1 and CUDA 12.1. Install requirements.txt in your environment. Obtain DINOv2 Base pretrained backbone weights separately and place them at pretrained/dinov2_base.pth. Obtain Cityscapes separately and prepare TrainId masks (0..18, ignore 255). Set data_root in configs/cityscapes_multisource.yaml.

## Train
```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 multisource_v2.py --config configs/cityscapes_multisource.yaml --labeled-id-path splits/cityscapes/1_16/labeled.txt --unlabeled-id-path splits/cityscapes/1_16/unlabeled.txt --save-path exp/cityscapes/multisource/base_1_16
```

Single-GPU settings: DINOv2 Base, crop 798, batch 1, BF16 and activation checkpointing, 40000 total steps. These differ from paper multi-image batch settings. latest.pth is saved every 500 steps and at epoch end. Resume reuses this file automatically. Within-epoch batch position is restored; data-worker RNG is not restored, so resumption is not bitwise deterministic. For teacher inference load model_ema and strip module. prefixes for a non-DDP model.

## Checks
```bash
python scripts/test_supervised_ce.py
PYTHONPATH=. python scripts/check_multisource_loss.py
```

All-ignore supervised crops yield a differentiable zero loss. Nonfinite outputs, losses and gradients stop training with diagnostic information. Historical NaNs were observed; the demonstrated all-ignore failure is repaired, but its relationship to every historical failure is not established. See FIX_NOTES.md.
