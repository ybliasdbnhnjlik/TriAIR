# TriAIR

**面向跨分辨率非对称食物图像检索的三层次知识蒸馏**

论文标题：**TriAIR: Triple-Level Knowledge Distillation for Cross-Resolution Asymmetric Food Image Retrieval**。

[English](README.md) · [数据准备](docs/DATASETS.md) · [配置说明](docs/CONFIGURATIONS.md) · [复现说明](docs/REPRODUCIBILITY.md)

TriAIR 使用低分辨率学生网络编码查询图像，使用冻结的高分辨率教师网络
编码图库。训练时结合三个蒸馏分支，以及配置中的交叉熵和 triplet 损失：

| 论文模块 | 作用 | 代码对应 |
| --- | --- | --- |
| RSD：关系结构蒸馏 | 传递教师的局部排序关系 | `d3_loss`、`D3.*` |
| DMGD：方向性中层引导蒸馏 | 对齐全局和语义有效局部特征，并沿教师降采样变化方向适度放松局部对齐 | `ugd_loss`、`UGD.RA_MODE: directional` |
| LSD：Logit 标准化蒸馏 | 在共享冻结教师分类器下对齐标准化类别分布 | `teacher_classifier_lsd_loss`、`UGD.LSD_*` |

核心实现位于 [TriAIR.py](AIR_Distiller/distillers/TriAIR.py)，类名和配置中的
`DISTILLER.TYPE` 统一使用 **`TriAIR`**。

TriAIR 的早期发布名称为 TriKD。已有配置中的 `DISTILLER.TYPE: TriKD`
及 `distillers.TriKD` 导入仍可使用；新配置采用 `TriAIR.yaml`，输出目录为
`outputs/TriAIR/`。

发布版提供 Food101、Food172、InShop、SOP 的 TriAIR 配置，保留 AIR-Distiller
对比方法，以及 CUB200、MSMT17 的原有基线配置。模型权重和数据集需单独准备。

## 安装

验证环境为 Python 3.10、PyTorch 2.4.1、torchvision 0.19.1、timm 1.0.28。
训练需要 CUDA GPU；评测和单元测试也支持 CPU。以下命令均在本目录执行。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

入口脚本自动处理模块导入路径，无需修改 Python 源码。

## 准备数据和教师

Food101 的默认数据根目录是 `data/food-101`，其下应有 `images/` 和
`meta/train.txt`、`meta/test.txt`。Food172 的默认根目录是 `data/food-172`，
其下应有 `vireoFood-172/`，并在其中放置图片、`train_full.txt`、`test_full.txt`。

Food101 每类按路径排序后的首张测试图像用作 query，其余用作 gallery；
Food172 使用固定划分种子 42，每类抽一张 query，其余用作 gallery。
Food172 的划分依赖标注顺序，应保留原始列表。InShop/SOP 需要预处理后的
AIR-Distiller 目录和文件名格式，详见[数据说明](docs/DATASETS.md)。

Food101 的 ResNet101 教师可用以下命令训练：

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food101/Vanilla/ResNet101_teacher.yaml

mkdir -p checkpoints/Food101
cp outputs/teachers/Food101_ResNet101_256x256/NONE_90.pth \
  checkpoints/Food101/ResNet101_256x256.pth
```

Food172 提供同结构配置，两个食物数据集均包含 Swin-V2-Small 教师配置。
这些教师配置来自已有监督训练记录；训练轮次与 stride 设置见[复现说明](docs/REPRODUCIBILITY.md)。
也可通过 `DISTILLER.TEACHER_MODEL_PATH` 指定已有教师权重，类别数和标签映射
必须与当前数据集一致。

## 训练学生

Food101：ResNet101（256×256）→ ResNet18（64×64）。

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food101/ResNet101_256x256_ResNet18_64x64/TriAIR.yaml \
  OUTPUT_DIR.EXPERIMENT_NAME food101_r101_r18
```

切换为 Food172 时，将配置路径中的 `Food101` 换成 `Food172`，并设置独立的
实验名称。其他网络组合和消融配置见[配置说明](docs/CONFIGURATIONS.md)。

支持通过命令行覆盖路径：

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/train.py \
  --cfg Training_Configs/Food101/ResNet101_256x256_ResNet18_64x64/TriAIR.yaml \
  DATASETS.ROOT_DIR /path/to/food-101 \
  DISTILLER.TEACHER_MODEL_PATH /path/to/teacher.pth \
  OUTPUT_DIR.EXPERIMENT_NAME food101_custom
```

训练默认使用 ImageNet 初始化，首次运行可能下载骨干权重。
`DISTILLER.STUDENT_PRETRAIN_PATH` 可指定兼容的本地预训练权重；设置
`DISTILLER.STUDENT_PRETRAIN_CHOICE False` 可从头训练学生。

`CUDA_VISIBLE_DEVICES` 优先于配置中的设备设置。多卡使用 DataParallel，
关系蒸馏在每张卡的局部 batch 内计算，因此 GPU 数量变化会改变候选集合。

## 评测与输出

上述 Food101 示例的输出目录为 `outputs/TriAIR/food101_r101_r18/`。

```bash
CUDA_VISIBLE_DEVICES=0 python AIR_Distiller/tools/test.py \
  --cfg outputs/TriAIR/food101_r101_r18/config.yaml
```

指定权重并在 CPU 评测：

```bash
python AIR_Distiller/tools/test.py \
  --cfg outputs/TriAIR/food101_r101_r18/config.yaml \
  --checkpoint outputs/TriAIR/food101_r101_r18/TriAIR_120.pth \
  EXPERIMENT.DEVICE cpu
```

| 输出 | 含义 |
| --- | --- |
| `config.yaml` | 包含命令行覆盖项的完整训练配置 |
| `TriAIR_120.pth` | 学生、教师及蒸馏模块的完整权重，供评测使用 |
| `student_120.pth` | 去除 `student.` 前缀的独立学生权重 |
| `train_log.txt`、`test_acc.txt` | 训练日志及检索评测记录 |
| `config.eval.yaml`、`test_log.txt` | 独立评测的配置与日志 |
| `inference_speed.txt` | 参数量与 MACs 估计，不包含实际延迟测量 |

完整权重已经包含教师，评测无需额外教师文件或 ImageNet 下载。
数据标注仍用于确定类别数。独立学生权重不能替代评测命令所需的完整权重。
权重文件不包含优化器和调度器状态，训练入口不提供精确断点续训。

## 测试

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

测试使用合成输入，检查损失与梯度、真实骨干接口、配置加载、权重保存与加载、
评测行为；无需下载数据集或训练权重。无 CUDA 时跳过对应 AMP 测试。

默认使用 NumPy 排序评测。如需编译可选的 Cython 加速实现：

```bash
python -m pip install -r requirements-optional.txt
(cd AIR_Distiller/utils/rank_cylib && python setup.py build_ext --inplace)
```

## 引用与许可

请引用配套 TriAIR 稿件，作者信息见 [CITATION.cff](CITATION.cff)。本次整理
未重新运行完整 benchmark，README 不引入新的性能结果。

本项目基于 [D3still / AIR-Distiller](https://github.com/SCY-X/D3still)，并保留
其上游来源说明。TriAIR 新增代码采用 [MIT](LICENSE) 许可。第三方许可及上游
D3still 缺失的版权声明事项记录在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
