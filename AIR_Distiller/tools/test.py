"""Evaluate a full distiller checkpoint with student queries and teacher galleries."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import build_distiller, configure_device, load_checkpoint, load_config, make_parser, prepare_output


def main():
    args = make_parser("TriKD asymmetric image retrieval evaluation", evaluation=True).parse_args()
    config = load_config(args)
    device = configure_device(config)

    from dataloader.make_dataloader import DataLoaderFactory
    from processor.inferencer import inference

    checkpoint = args.checkpoint or str(
        Path(config.OUTPUT_DIR.ROOT_PATH) / config.OUTPUT_DIR.EXPERIMENT_NAME
        / f"{config.DISTILLER.TYPE}_{config.TEST.WEIGHT}.pth"
    )
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Use --checkpoint or set OUTPUT_DIR and TEST.WEIGHT.")
    prepare_output(config, args, training=False)
    query_loader, gallery_loader, num_classes = DataLoaderFactory(config).create_evaluation_dataloaders()
    distiller = build_distiller(config, num_classes, training=False)
    # Full checkpoints contain the teacher and every distillation module.
    # Evaluation needs neither ImageNet downloads nor a separate teacher file.
    distiller.load_state_dict(load_checkpoint(checkpoint))
    inference(config, distiller.to(device), query_loader, gallery_loader)


if __name__ == "__main__":
    main()
