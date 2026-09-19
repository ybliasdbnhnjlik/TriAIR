# Release validation

## Paper and project rename — 2026-09-19

The paper title and project name were updated to **TriAIR: Triple-Level
Knowledge Distillation for Cross-Resolution Asymmetric Food Image Retrieval**.
The checks below use the Python 3.10 environment recorded in the original
release validation.

- All 42 existing unit tests passed with CUDA available and the NumPy evaluator;
  this includes loading all 199 public configurations.
- Legacy `TriKD` imports and the `DISTILLER.TYPE: TriKD` identifier resolve to
  the same implementation as `TriAIR`.
- The English and Chinese READMEs and `CITATION.cff` match the manuscript title.
- Python syntax and local documentation links were checked after the file renames.

## Original release — 2026-09-16

The release was checked with Python 3.10.20, PyTorch 2.4.1, torchvision 0.19.1,
timm 1.0.28 and NumPy 2.2.6. Runtime versions are recorded in `requirements.txt`.

| Check | Result |
| --- | --- |
| Public YAML files merged with YACS; dataset/model/trainer registrations checked | 199 configurations passed |
| Complete unit suite with CUDA available and the compiled evaluator | 42 tests passed, including CUDA AMP tests |
| Unit suite after TriKD naming unification, using the NumPy evaluator with CUDA available | 42 tests passed; all 199 configurations load with the current registrations |
| Release runtime suite on CPU with only the NumPy evaluator | 11 tests passed |
| Optional rank/ROC extension build using Cython 3.3.0 | Passed |
| Supervised ResNet101 teacher training and standalone evaluation | Passed |
| ResNet101→ResNet18 TriKD training, full-checkpoint save and standalone evaluation | Passed |
| CPU flip evaluation after removing the original teacher checkpoint | Passed |
| Raw exported student weights loaded into a standalone ResNet18 | Passed; 512-dimensional retrieval output |
| Core TriKD, D3 and UGD code compared against the research source | Computation unchanged; public class/module naming and comments/docstrings updated |
| Inherited configuration hyperparameters compared against the research source | Numeric values unchanged; deployment paths, method/experiment names and device selection normalized |
| Source tree and local documentation links | Checked; no machine paths, weights, caches or generated native binaries |

The end-to-end run used synthetic RGB images: four classes, 16 training images,
four queries and eight gallery images, at 64×64 resolution for one epoch.
It checks command/config/checkpoint integration and does not measure benchmark
accuracy. The final-epoch checkpoint was also verified with a checkpoint interval
larger than the run length. Synthetic data and generated checkpoints are not included.

The native binaries used for validation were removed from the release. Users
may compile the optional evaluator locally, or use the default NumPy evaluator.
The GitHub Actions workflow is included; it has not been run on GitHub as part
of this local validation.
