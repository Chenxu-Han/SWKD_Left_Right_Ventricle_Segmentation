# SWKD

PyTorch implementation of **Decoupling Representation Learning from Task Learning: Stage-Wise Knowledge Distillation for Echocardiography Segmentation**.

SWKD distills a frozen SAM teacher into DeepLabV3-MobileNetV2, Mobile-UNet, or FSSNet for left and right ventricular segmentation. Stage 1 trains the encoder with frequency-perturbed images and teacher features. Stage 2 freezes the encoder and trains the decoder with segmentation supervision and uncertainty-aware distillation.

## Code structure

```text
Code_SWKD_Echocardiography_Segmentation/
├── dataset/
│   └── datasets.py                 # LV/RV images, masks and split lists
├── models/
│   ├── base_models/                # Included upstream implementations
│   │   ├── SAM/
│   │   ├── DeepLabV3_MobileNetV2/
│   │   ├── Mobile_UNet/
│   │   └── FSSNet/
│   ├── base.py                     # Encoder/decoder training stages
│   ├── sam.py                      # Frozen SAM teacher and point prompts
│   ├── deeplabv3.py                # DeepLabV3-MobileNetV2 student
│   ├── mobile_unet.py              # Mobile-UNet student
│   ├── fssnet.py                   # FSSNet student
│   └── model_zoo.py                # Student selection
├── losses/
│   ├── frd.py                      # Frequency reconstruction denoising
│   ├── uaw.py                      # Uncertainty-aware weighting
│   └── segmentation.py             # Cross-entropy and foreground Dice
├── utils/
│   ├── checkpoint.py              # Weight loading, saving and resume
│   ├── metrics.py                 # DSC and HD95
│   └── train_utils.py             # Seed, training log and previews
├── weights/
├── config.py
├── train.py
├── test.py
├── infer.py
├── requirements.txt
└── README.md
```

## Installation

The code has been tested with Python 3.8.19, PyTorch 2.4.1, and torchvision 0.19.1 on CPU. Create a separate environment to avoid conflicts with packages from other projects:

```bash
conda create -n swkd python=3.8.19
conda activate swkd
```

Install PyTorch 2.4.1 and torchvision 0.19.1 for your CUDA version before installing the remaining dependencies.

From the repository directory:

```bash
python -m pip install -r requirements.txt
```

SAM and Segmentation Models PyTorch are installed from these local directories; the Mobile-UNet and FSSNet adapters import their included implementations directly. Reinstall the dependencies after moving the repository.

## Weights

Download the official [SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `weights/sam_vit_b_01ec64.pth`. Trained student weights are not provided.

The official checkpoint is pretrained on SA-1B. To reproduce the paper's task-adapted teacher experiments, supply the corresponding fine-tuned SAM checkpoint with `--teacher`.

## Training

```bash
python train.py --teacher weights/sam_vit_b_01ec64.pth --train-list data/LVRV-train.txt --val-list data/LVRV-val.txt --student DeepLabV3_MobileNetV2 --run-dir runs/deeplab
```

Use `--student Mobile_UNet` or `--student FssNet` for the other students. This runs both training stages. Training and distillation defaults are in [config.py](config.py).

```bash
python train.py --teacher weights/sam_vit_b_01ec64.pth --train-list data/LVRV-train.txt --val-list data/LVRV-val.txt --student DeepLabV3_MobileNetV2 --run-dir runs/deeplab --resume runs/deeplab/stage2/last.pth
```

Stage 2 fixes the encoder, including batch normalization statistics. Teacher output distillation uses one positive point per organ for images with nonempty LV and RV masks; all training images receive segmentation supervision.
UAW is averaged over the full batch. Checkpoints record this reduction; older checkpoints using an eligible-sample average remain loadable for inference but require their original training code to resume.

## Evaluation

```bash
python test.py --weights runs/deeplab/stage2/best.pth --test-list data/LVRV-test.txt --output results/deeplab
```

Use `--resolution original` for evaluation at the original image size, or `--external` for datasets outside the seven primary datasets. The original-resolution results use a different evaluation grid and should be reported separately.

## Inference

```bash
python infer.py --weights runs/deeplab/stage2/best.pth --input data/images --output predictions/deeplab
```

The input can be an image, a directory, or a list file. Inference saves three-class label PNGs and binary LV/RV masks at the original image size under `--output`. Only the student checkpoint is required. For a raw state dictionary, also specify `--student`.

## Citation

Please cite **Decoupling Representation Learning from Task Learning: Stage-Wise Knowledge Distillation for Echocardiography Segmentation** when using this code.
