# Configurations

All YAML files inherit [defaults.py](../AIR_Distiller/config/defaults.py).
Use `--cfg <file>` followed by optional `KEY VALUE` pairs. Effective settings
are saved with each run. Training and evaluation use the same method identifier.

## TriAIR

The main configurations are `Training_Configs/{Food101,Food172,InShop,SOP}/<pair>/TriAIR.yaml`.
Each dataset provides ResNet101→ResNet18, ResNet101→MobileNetV3-Small and
Swin-V2-Small→ResNet18, at teacher/student resolutions 256×256 / 64×64.

| Setting | Meaning |
| --- | --- |
| `DISTILLER.TYPE: TriAIR` | Instantiate TriAIR. |
| `SOLVER.TRAINER: kd` | Use supervised and distillation loaders. |
| `D3.TOPK` | Teacher-neighborhood size for RSD, including the first entry. |
| `D3.ALPHA/BETA/GAMMA` | Feature, hard-order and easy-order terms inside RSD. |
| `D3.KD_WEIGHT` | Overall RSD weight. |
| `UGD.ALPHA` | Global feature-alignment weight inside DMGD. |
| `UGD.BETA` | Weight shared by the two local granularities. |
| `UGD.KD_WEIGHT` | Overall DMGD weight. |
| `UGD.RA_ENABLED: true` | Enable the paired degradation branch. |
| `UGD.RA_MODE: directional` | Select the paper's directional local distance. |
| `UGD.RA_DIRECTION_RHO` | Directional relaxation strength; 0 means aligned L2, valid range `[0,1)`. |
| `UGD.RA_USE_CLASS_MASK` | Select regions whose teacher class prediction matches the image label. |
| `UGD.LSD_WEIGHT`, `UGD.LSD_TAU` | LSD coefficient and temperature. |
| `INPUT.KD_SYNC_AUGMENTATION: false` | Independent main KD views; DMGD constructs its aligned local view internally. |

`UGD.RA_*` and `D3_*` are historical implementation names. They do not introduce
extra paper components. Legacy scalar-RA options remain loadable for older
configurations; the main TriAIR YAML files explicitly select directional mode.

`loss_kd` already contains the weighted RSD and DMGD terms. `loss_kd_d3` and
`loss_kd_ugd` are diagnostics and must not be added to the objective again.
LSD is returned as `loss_LS_KD` and is added separately by the trainer.

## Ablations

Food101 and InShop provide ablations under the ResNet101→ResNet18 configuration.
Their existing filenames use `MGD`/`RA_UGD`; with the current settings these
refer to directional DMGD.

| Filename | Global alignment | RSD | Local DMGD | LSD |
| --- | --- | --- | --- | --- |
| `TriAIR_A1_Lf.yaml` | On | Off | Off | Off |
| `TriAIR_A2_Lf_RSD.yaml` | On | On | Off | Off |
| `TriAIR_A2_Lf_RA_UGD.yaml` | On | Off | On | Off |
| `TriAIR_A3_Lf_RSD_MGD.yaml` | On | On | On | Off |
| `TriAIR_A4_Lf_RSD_MGD_LSD.yaml` | On | On | On | On |

Cross-entropy and triplet settings are inherited from the corresponding YAML.
The ablations disable losses through their coefficients; they do not promise
to eliminate the computation of every disabled branch.

## Teachers and baselines

Food101 and Food172 include `Vanilla/ResNet101_teacher.yaml` and
`Vanilla/Swin_Transformer_V2_Small_teacher.yaml`, extracted from saved supervised
training configurations. The InShop/SOP/CUB200/MSMT17 `Vanilla/` folders and
per-method YAML files are inherited baselines. Food datasets do not contain a
complete per-method baseline sweep.

To train an undistilled model use `DISTILLER.TYPE: NONE` and
`SOLVER.TRAINER: vanilla`. To evaluate its full checkpoint, keep the same
settings; the model is used for both query and gallery.

Student model options include `ResNet18`, `MobileNetV3_Small` and the other
entries in [model_dict](../AIR_Distiller/models/__init__.py). Compatible teacher
and student feature interfaces are required when introducing a new pair.
