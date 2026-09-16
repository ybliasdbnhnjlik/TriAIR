# Dataset preparation

`DATASETS.ROOT_DIR` is passed directly to the selected loader. No data is
downloaded automatically. Dataset files and checkpoints are excluded from the release.

| `DATASETS.NAMES` | Default root | Loader appends | Required inputs |
| --- | --- | --- | --- |
| `Food101` | `./data/food-101` | Nothing | `images/`, `meta/train.txt`, `meta/test.txt` |
| `Food172` | `./data/food-172` | `vireoFood-172/` | `train_full.txt`, `test_full.txt`, images |
| `InShop` | `./data` | `InShop/` | `train/`, `query/`, `gallery/` |
| `SOP` | `./data` | `Stanford_Online_Products/` | `train/`, `test/` |
| `CUB200` | `./data` | `CUB_200_2011/` | `train/`, `test/` |
| `MSMT17` | `./data` | `MSMT17/` | `train/`, `query/`, `gallery/` |

## Food101

Keep the Food-101 image and metadata structure. Each annotation line is an
image path relative to `images/`, **without** the `.jpg` extension:

```text
apple_pie/1005649
baby_back_ribs/1005293
```

The loader maps training class names to integers in sorted order. For retrieval,
it groups `meta/test.txt` by class, sorts image paths within each class, and takes
the **first image per class as query**, with all remaining test images as gallery.
Query camera IDs are 0 and gallery camera IDs are 1.

With the standard Food-101 metadata, this produces 75,750 training images,
101 queries and 25,149 gallery images. The query/gallery split is implemented
by this repository; it is not an additional official Food-101 split. Each test
class must contain at least two images and the train/test class sets must match.

## Food172

The exact folder name expected by the loader is `vireoFood-172`:

```text
data/food-172/vireoFood-172/
├── train_full.txt
├── test_full.txt
└── <images addressed by the annotation files>
```

Each annotation line contains a relative image path, including its extension,
followed by an integer class label:

```text
1/example.jpg 1
2/another-example.jpg 2
```

Paths must be relative to `vireoFood-172/`. The prepared `train_full.txt` and
`test_full.txt` files must be supplied alongside the images. The release does
not reconstruct these lists from another dataset distribution.

The loader selects **one random query per class using split seed 42**, then
uses the remaining test images as gallery. This is separate from the training
seed `SOLVER.SEED`, normally 2024. Selection depends on the annotation order;
keep `test_full.txt` unchanged when comparing runs. Training labels are remapped
in sorted string order, whereas evaluation keeps the integer labels from the
annotations. Each test class must contain at least two images.

## InShop, SOP, CUB200 and MSMT17

These inherited loaders expect the **prepared AIR-Distiller layouts**, not
the unmodified archives from the original datasets. Images are directly inside
each split directory, not in nested class folders.

| Loader | Extension | Filename fields used by the loader | Evaluation split |
| --- | --- | --- | --- |
| InShop | `.jpg` | Item ID before the first underscore, e.g. `17_example.jpg` | Separate `query/` and `gallery/`; camera IDs 0 and 1 |
| SOP | `.JPG` | Two integer fields `identity_image.JPG` | `test/` serves as both query and gallery; self-matches are excluded |
| CUB200 | `.jpg` | Two integer fields `identity_image.jpg` | `test/` serves as both query and gallery; self-matches are excluded |
| MSMT17 | `.jpg` | Identity and camera as `identity_c<camera>_...jpg` | Separate `query/` and `gallery/` |

Consult the [dataset loader sources](../AIR_Distiller/dataloader/datasets)
before converting another distribution. Preserve the original training/test
identities and query/gallery membership. The upstream
[AIR-Distiller repository](https://github.com/SCY-X/D3still) documents its prepared
data and teacher distributions; those external files are not bundled here.

## Class counts and checkpoints

The classifier size is inferred from the training split, even for evaluation.
A teacher checkpoint must use the same class count and class-label mapping as
the current training metadata. Changing the number or ordering of classes
requires a matching teacher; loading a checkpoint alone cannot recover its
original label mapping.
