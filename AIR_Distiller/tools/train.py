"""Train a supervised teacher/student or a TriKD distillation model."""

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.common import build_distiller, configure_device, load_config, make_parser, prepare_output


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Preserve the research training settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args = make_parser("TriKD asymmetric image retrieval training").parse_args()
    config = load_config(args)
    device = configure_device(config, training=True)

    import torch
    from dataloader.make_dataloader import DataLoaderFactory
    from processor import trainer_dict

    set_seed(config.SOLVER.SEED)
    output_dir = prepare_output(config, args, training=True)
    loaders = DataLoaderFactory(config).create_dataloaders()
    train_loader, kd_loader, query_loader, gallery_loader, num_classes = loaders
    distiller = build_distiller(config, num_classes, training=True).to(device)

    base_params = distiller.get_base_parameters()
    extra_params = distiller.get_extra_parameters()
    base_macs = distiller.get_base_flops(config.INPUT.STUDENT_SIZE_TRAIN)
    with (output_dir / "inference_speed.txt").open("a") as stream:
        stream.write(f"Student parameters: {base_params:.3f} M\n")
        stream.write(f"Distillation parameters: {extra_params:.3f} M\n")
        stream.write(f"Student MACs (ptflops): {base_macs}\n")
    print(f"Student: {base_params:.3f} M parameters, {base_macs} MACs")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        distiller = torch.nn.DataParallel(distiller)

    trainer = trainer_dict[config.SOLVER.TRAINER](
        config, distiller, train_loader, kd_loader, query_loader, gallery_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
