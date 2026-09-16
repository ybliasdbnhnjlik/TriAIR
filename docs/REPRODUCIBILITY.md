# Reproducibility and release scope

This release uses `TriKD` for the implementation class, method identifier and
configuration filenames, with directional RA and teacher-classifier LSD settings.
Core loss formulas, default weights,
backbone feature interfaces and existing query/gallery protocols are preserved.
Full benchmark training has not been repeated as part of the code cleanup;
no new retrieval accuracy claims are introduced in the README.

## Training protocol

- The supervised loader defines an epoch. The KD loader is restarted when it
  runs out, so its batch size does not truncate supervised training.
- Main supervised and KD views use the configured augmentations. The local
  directional branch constructs aligned student/teacher proxy views internally.
- The teacher is frozen and stays in evaluation mode. The auxiliary student
  branch preserves its established BatchNorm buffer handling.
- CUDA seeds and the existing cuDNN settings are retained. The training seed
  does not imply bitwise reproducibility across hardware or software versions.
- `DataParallel` computes pairwise distillation on each replica's local batch.
  Changing the number of GPUs therefore changes the relational candidate pool.
- A triplet training batch must be divisible by `DATALOADER.NUM_INSTANCE`.
  The KD batch must be large enough to contain the intended `D3.TOPK` neighbors.
  Each loader must yield at least one batch.

## Food teacher settings

The supplied food ResNet101 teachers use 90 epochs and the Swin-V2-Small
teachers use 25 epochs, matching the saved supervised YAML files. The
ResNet101 supervised files set `STUDENT_LAST_STRIDE: 2`; the main TriKD
configurations inherit `TEACHER_LAST_STRIDE: 1` when loading the teacher, as
in the supplied research implementation. These values are exposed rather
than silently unified. The weight shapes are compatible, but the stride
changes feature-map resolution and must be recorded in experimental comparisons.

Checkpoint files contain weights only. Exact continuation of a previous
optimizer/scheduler run is not supported by the train CLI. The standalone
test CLI uses the full distiller checkpoint, while the exported
`student_<epoch>.pth` contains raw student weights for deployment.

## Changes for public use

- Portable relative paths and shared train/test configuration handling.
- Effective training and evaluation configs saved separately.
- Consistent CPU-mapped teacher/full-checkpoint loading, including legacy prefixes.
- Evaluation initializes models directly from its full checkpoint, with no
  separate teacher or ImageNet download.
- Evaluation does not construct training samplers, and supports CPU and feature
  flipping with the actual embedding dimension.
- Multi-GPU wrapper handling corrected in profiling, saving, optimization and evaluation.
- Final-epoch checkpoints are saved even when the checkpoint interval does not
  divide the epoch count. Student exports have their wrapper prefix removed.
- Filename metadata is parsed from filenames rather than parent directory names.
- Fixed one inherited VanillaKD YAML indentation error and the multi-step
  scheduler's logging key. Removed the obsolete NumPy boolean alias and unused
  FAISS imports from ranking utilities.

The source distribution contains no logs, trained weights, native binaries,
Python caches, manuscript drafts, internal reviews or D3_LEG follow-up experiments.

## Validation

Run `OMP_NUM_THREADS=1 python -m unittest discover -s tests -v`.
Tests cover loss values/gradients, AMP, BatchNorm behavior, real backbone
interfaces, all public configurations, checkpoint exports and portable evaluation.
They use synthetic inputs, and CUDA-only tests skip on CPU installations.
The optional native evaluator must be compiled locally for the current Python
and NumPy ABI; no precompiled `.so` files are distributed.

The completed local checks and their scope are recorded in [VALIDATION.md](VALIDATION.md).
