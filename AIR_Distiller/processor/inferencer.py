"""Retrieval evaluation shared by training and the standalone test command."""

import logging
from pathlib import Path

import torch

from utils.metrics import R1_mAP_eval


@torch.no_grad()
def inference(cfg, distiller, query_loader, gallery_loader):
    logger = logging.getLogger("Asymmetric_Image_Retrieval.test")
    # DataParallel does not forward custom methods such as forward_query.
    model = distiller.module if isinstance(distiller, torch.nn.DataParallel) else distiller
    device = next(model.parameters()).device
    model.eval()
    evaluator = R1_mAP_eval(
        max_rank=100, metric=cfg.TEST.TEST_METRIC,
        reranking=cfg.TEST.RE_RANKING,
        reranking_parameters=cfg.TEST.RE_RANKING_PARAMETER,
    )

    for loader, forward, update in (
        (query_loader, model.forward_query, evaluator.query_update),
        (gallery_loader, model.forward_gallery, evaluator.gallery_update),
    ):
        for images, pids, camids in loader:
            images = images.to(device)
            features = forward(image=images)
            if cfg.TEST.FLIP_FEATS == "on":
                features = features + forward(image=images.flip(-1))
            update((features.cpu(), pids, camids))

    cmc, mean_ap, mean_inp = evaluator.compute()
    ranks = ", ".join(f"Rank-{rank}: {cmc[rank - 1]:.2%}"
                      for rank in (1, 5, 10) if rank <= len(cmc))
    result = f"mAP: {mean_ap:.2%}, mINP: {mean_inp:.2%}, {ranks}"
    logger.info("Validation results: %s", result)
    output_dir = Path(cfg.OUTPUT_DIR.ROOT_PATH) / cfg.OUTPUT_DIR.EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_acc.txt").open("a") as stream:
        stream.write(f"[Re-ranking: {cfg.TEST.RE_RANKING}] {result}\n")
    return cmc, mean_ap, mean_inp
