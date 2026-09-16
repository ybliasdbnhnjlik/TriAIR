# Third-party code and attribution

TriKD extends [SCY-X/D3still](https://github.com/SCY-X/D3still), including its
AIR-Distiller framework and D3still/UGD implementations. This release retains
the upstream module names and source-level credits. TriKD's additions are
covered by [LICENSE](LICENSE); that notice does not replace upstream ownership.

The upstream D3still README declares MIT licensing. On 2026-09-16, its linked
root `LICENSE` file returned HTTP 404. A separate D3still copyright notice
could therefore not be recovered. The upstream statement is recorded here,
without inventing an upstream copyright holder or claiming a complete license
audit. Maintainers should obtain that notice from upstream before public release.

The upstream framework acknowledges
[MEGVII mdistiller / DKD](https://github.com/megvii-research/mdistiller).
Its [MIT notice](licenses/mdistiller-MIT.txt) is included verbatim.

Retrieval evaluation files credit
[Kaiyang Zhou's deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).
Its [MIT notice](licenses/deep-person-reid-MIT.txt) is included verbatim.

Other original source references remain in the corresponding files, including
PKT, Open-ReID triplet mining, and k-reciprocal re-ranking. Backbone weights
come from PyTorch/torchvision, timm, or IBN-Net when downloaded by their respective
model loaders. No third-party datasets or model weights are distributed here;
their providers' terms apply separately.
