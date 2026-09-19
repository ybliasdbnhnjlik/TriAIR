# TriAIR

**Triple-Level Knowledge Distillation for Cross-Resolution Asymmetric Food Image Retrieval**

[中文说明](README.zh-CN.md) · [Dataset preparation](docs/DATASETS.md) · [Configurations](docs/CONFIGURATIONS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

TriAIR trains a lightweight, low-resolution **query student** to retrieve images
from a gallery encoded by a frozen, high-resolution **teacher**. It combines
three supervision branches:

| Paper component | Implementation | Purpose |
| --- | --- | --- |
| Relational Structure Distillation (RSD) | `d3_loss`, `D3.*` | Transfer the teacher's local ranking structure. |
| Directional Middle Level Guidance Distillation (DMGD) | `ugd_loss`, `UGD.RA_MODE: directional` | Align global and selected local features, relaxing local alignment along the teacher's downsampling direction. |
| Logit-Standardization Distillation (LSD) | `teacher_classifier_lsd_loss`, `UGD.LSD_*` | Align standardized class distributions using the same frozen teacher classifier. |

The implementation class and `DISTILLER.TYPE` identifier are **`TriAIR`**.
The main code is
[AIR_Distiller/distillers/TriAIR.py](AIR_Distiller/distillers/TriAIR.py).
Training also uses the configured cross-entropy and triplet losses.

TriAIR was originally released as TriKD. Saved configurations using
`DISTILLER.TYPE: TriKD` and imports from `distillers.TriKD` remain supported.
New configurations use `TriAIR.yaml` and write to `outputs/TriAIR/`.

This release contains TriAIR configurations for **Food101, Food172, InShop and
SOP**, together with the AIR-Distiller comparison methods: VanillaKD, FitNet,
CC, RKD, PKT, CSD, ROP, RAML, D3 and UGD. Additional CUB200/MSMT17 configurations
are inherited baselines. Datasets and trained checkpoints are supplied separately.

## Installation

Run commands from the repository root. The validated environment uses Python
3.10, PyTorch 2.4.1, torchvision 0.19.1 and timm 1.0.28. Training requires a
CUDA-capable GPU. Evaluation and unit tests also support CPU.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python AIR_Distiller/tools/train.py --help
python AIR_Distiller/tools/test.py --help
```

The scripts resolve Python imports automatically. Relative data, checkpoint
and output paths are interpreted from the working directory.

## Prepare data and teacher checkpoints

The food datasets use different root conventions:

```text
data/
├── food-101/
│   ├── images/<class>/<image>.jpg
│   └── meta/{train,test}.txt
├── food-172/
│   └── vireoFood-172/
│       ├── <relative image paths>
│       ├── train_full.txt
│       └── test_full.txt
├── InShop/{train,query,gallery}/
└── Stanford_Online_Products/{train,test}/
```

Follow [dataset preparation](docs/DATASETS.md) for annotation formats, filename
requirements, and the exact query/gallery splits. These loaders expect data
that has already been prepared; they do not download or convert datasets.

Teacher checkpoint locations are configured by `DISTILLER.TEACHER_MODEL_PATH`.
The Food101 ResNet101 example expects
`checkpoints/Food101/ResNet101_256x256.pth`. It accepts either a raw model
`state_dict` or a supervised checkpoint with `student.`-prefixed keys.

To train this teacher using the supplied supervised configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food101/Vanilla/ResNet101_teacher.yaml

mkdir -p checkpoints/Food101
cp outputs/teachers/Food101_ResNet101_256x256/NONE_90.pth \
  checkpoints/Food101/ResNet101_256x256.pth
```

Food172 has the same teacher configuration layout. Swin-V2-Small teacher
configurations are also included. Their supervised schedules and stride settings
come from the saved research configurations; see [reproducibility notes](docs/REPRODUCIBILITY.md).

ImageNet initialization is enabled for training by default and may download
backbone weights on the first run. Set `DISTILLER.STUDENT_PRETRAIN_PATH` to a
compatible local ImageNet checkpoint, or
`DISTILLER.STUDENT_PRETRAIN_CHOICE False` to train from scratch. A trained
retrieval teacher is required for distillation.

## Train TriAIR

Food101, ResNet101 teacher at 256×256 → ResNet18 student at 64×64:

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food101/ResNet101_256x256_ResNet18_64x64/TriAIR.yaml \
  OUTPUT_DIR.EXPERIMENT_NAME food101_r101_r18
```

Food172 uses the corresponding config:

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food172/ResNet101_256x256_ResNet18_64x64/TriAIR.yaml \
  OUTPUT_DIR.EXPERIMENT_NAME food172_r101_r18
```

Override paths and settings by appending `KEY VALUE` pairs after `--cfg`:

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food101/ResNet101_256x256_ResNet18_64x64/TriAIR.yaml \
  DATASETS.ROOT_DIR /path/to/food-101 \
  DISTILLER.TEACHER_MODEL_PATH /path/to/teacher.pth \
  OUTPUT_DIR.EXPERIMENT_NAME food101_custom
```

`CUDA_VISIBLE_DEVICES` takes priority over the configuration's device selection.
Multiple visible GPUs use `DataParallel`; a single GPU is the reference setup
because relational distillation uses the batch available on each replica.

For the Food101 example, outputs are written to `outputs/TriAIR/food101_r101_r18/`:

| File | Contents |
| --- | --- |
| `config.yaml` | Effective training configuration, including command-line overrides. |
| `TriAIR_120.pth` | Full distiller: student, teacher, and distillation modules. |
| `student_120.pth` | Raw student state dictionary, with no `student.` prefix. |
| `train_log.txt`, `test_acc.txt` | Training log and retrieval evaluations. |
| `inference_speed.txt` | Parameter counts and ptflops MAC estimate; this is not a measured latency benchmark. |

Checkpoints are saved at `SOLVER.CHECKPOINT_PERIOD` and at the final epoch.
They contain model weights, not optimizer/scheduler state for exact training resumption.

## Evaluate

Evaluate the full checkpoint produced above:

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/test.py \
  --cfg outputs/TriAIR/food101_r101_r18/config.yaml
```

For another checkpoint or CPU evaluation:

```bash
python AIR_Distiller/tools/test.py \
  --cfg outputs/TriAIR/food101_r101_r18/config.yaml \
  --checkpoint outputs/TriAIR/food101_r101_r18/TriAIR_120.pth \
  EXPERIMENT.DEVICE cpu
```

Evaluation loads the **full distiller checkpoint** directly and needs no separate
teacher file or ImageNet download. The dataset metadata is still required to
construct the models with the correct class count. `student_120.pth` is for
standalone student use and cannot replace the full checkpoint in this command.
Evaluation writes `config.eval.yaml`, `test_log.txt` and `test_acc.txt`.

The evaluator reports mAP, mINP and available Rank-1/5/10 values. Use mAP and
Rank-1 for the principal retrieval comparison. Flip evaluation can be enabled
with `TEST.FLIP_FEATS on`; re-ranking is disabled by default.

## Configurations and ablations

Each of the four TriAIR datasets provides these teacher/student pairs:

| Configuration directory | Teacher | Student |
| --- | --- | --- |
| `ResNet101_256x256_ResNet18_64x64` | ResNet101 | ResNet18 |
| `ResNet101_256x256_MobileNetV3_64x64` | ResNet101 | MobileNetV3-Small |
| `Swin_Transformer_V2_Small_256x256_ResNet18_64x64` | Swin-V2-Small | ResNet18 |

Use `Training_Configs/<dataset>/<pair>/TriAIR.yaml` for TriAIR. Food101 and
InShop include component ablations. See [configuration details](docs/CONFIGURATIONS.md)
for loss weights, inherited filenames and baseline coverage.

## Tests

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests cover the distillation losses and gradients, teacher feature interfaces,
all public YAML configurations, checkpoint round-trips and retrieval evaluation.
CUDA-specific tests are skipped when CUDA is unavailable. Tests use synthetic
inputs and do not require dataset downloads or trained weights.
Completed release checks are recorded in [the validation report](docs/VALIDATION.md).

The NumPy ranking evaluator works without compilation. To enable the optional
Cython implementation on your own machine:

```bash
python -m pip install -r requirements-optional.txt
(cd AIR_Distiller/utils/rank_cylib && python setup.py build_ext --inplace)
```

## Repository layout

```text
AIR_Distiller/
├── config/       # YACS defaults
├── dataloader/   # Dataset protocols, transforms, samplers
├── distillers/   # TriAIR and comparison methods
├── models/       # Backbones and retrieval heads
├── processor/    # Training and retrieval evaluation
├── solver/       # Optimizers and learning-rate schedules
├── tools/        # Train/test CLIs and shared runtime helpers
└── utils/        # Metrics, logging, optional native ranking code
Training_Configs/ # TriAIR, teacher, baseline and ablation YAML files
docs/            # Data and reproducibility documentation
tests/           # Synthetic regression tests
licenses/        # Preserved upstream license notices
```

## Citation and acknowledgements

Please cite the accompanying **TriAIR: Triple-Level Knowledge Distillation
for Cross-Resolution Asymmetric Food Image Retrieval** manuscript. Author metadata from the
manuscript is recorded in [CITATION.cff](CITATION.cff); a publication DOI has
not been supplied with this release.

TriAIR builds on [D3still / AIR-Distiller](https://github.com/SCY-X/D3still),
including its D3still and UGD implementations. The upstream framework also
acknowledges [mdistiller / DKD](https://github.com/megvii-research/mdistiller).
Original TriAIR contributions use the [MIT license](LICENSE). Upstream attribution
and the outstanding D3still license-notice item are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
