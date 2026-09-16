"""Shared configuration, device setup and checkpoint loading for the CLIs."""

import argparse
import os
from pathlib import Path

from config import cfg as defaults


def make_parser(description, evaluation=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", required=True, help="Path to a YAML configuration")
    if evaluation:
        parser.add_argument("--checkpoint", help="Full distiller checkpoint; overrides TEST.WEIGHT")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional KEY VALUE configuration overrides")
    return parser


def load_config(args):
    config = defaults.clone()
    with open(args.cfg, encoding="utf-8") as stream:
        incoming = defaults.load_cfg(stream)
    # YACS literal-evaluates strings on merge: a dumped GPU ID '0' otherwise
    # becomes integer 0 and cannot be merged back into its string default.
    if "EXPERIMENT" in incoming and "DEVICE_ID" in incoming.EXPERIMENT:
        incoming.EXPERIMENT.DEVICE_ID = repr(str(incoming.EXPERIMENT.DEVICE_ID).strip(chr(39) + chr(34)))
    config.merge_from_other_cfg(incoming)
    options = list(args.opts or [])
    for index in range(0, len(options) - 1, 2):
        if options[index] == "EXPERIMENT.DEVICE_ID":
            options[index + 1] = repr(options[index + 1].strip(chr(39) + chr(34)))
    config.merge_from_list(options)
    config.freeze()
    return config


def configure_device(config, training=False):
    # Set visibility before the first CUDA operation; a shell setting takes priority.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.EXPERIMENT.DEVICE_ID.strip(chr(39) + chr(34)))
    import torch

    device = config.EXPERIMENT.DEVICE
    if device not in {"cuda", "cpu"}:
        raise ValueError("EXPERIMENT.DEVICE must be 'cuda' or 'cpu'")
    if training and device != "cuda":
        raise ValueError("Training requires CUDA; CPU is supported for evaluation and unit tests.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check PyTorch and CUDA_VISIBLE_DEVICES, or use EXPERIMENT.DEVICE cpu for evaluation.")
    return torch.device(device)


def prepare_output(config, args, training):
    from utils.logger import setup_logger

    output_dir = Path(config.OUTPUT_DIR.ROOT_PATH) / config.OUTPUT_DIR.EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save effective settings, including CLI overrides. Evaluation has its own file.
    name = "config.yaml" if training else "config.eval.yaml"
    (output_dir / name).write_text(config.dump(), encoding="utf-8")
    logger = setup_logger("Asymmetric_Image_Retrieval", str(output_dir), if_train=training)
    logger.info("Arguments: %s", args)
    logger.info("Running with config:\n%s", config)
    return output_dir


def load_checkpoint(path):
    """Read a raw or state_dict-wrapped checkpoint on CPU."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_teacher_checkpoint(teacher, path):
    """Accept a raw teacher or the student branch of a supervised NONE checkpoint."""
    state = load_checkpoint(path)
    if any(key.startswith("student.") for key in state):
        state = {key.removeprefix("student."): value for key, value in state.items()
                 if key.startswith("student.")}
    teacher.load_state_dict(state)


def build_distiller(config, num_classes, training):
    from distillers import distiller_dict
    from models import model_dict

    teacher = None
    if config.DISTILLER.TYPE != "NONE":
        if training:
            path = config.DISTILLER.TEACHER_MODEL_PATH
            if not path or not Path(path).is_file():
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {path!r}. Set DISTILLER.TEACHER_MODEL_PATH to a trained teacher."
                )
        teacher = model_dict[config.DISTILLER.TEACHER_NAME](
            pretrained=False,
            last_stride=config.DISTILLER.TEACHER_LAST_STRIDE,
            num_classes=num_classes,
        )
        if training:
            load_teacher_checkpoint(teacher, config.DISTILLER.TEACHER_MODEL_PATH)

    pretrained = training and config.DISTILLER.STUDENT_PRETRAIN_CHOICE
    student = model_dict[config.DISTILLER.STUDENT_NAME](
        pretrained=pretrained,
        pretrained_path=config.DISTILLER.STUDENT_PRETRAIN_PATH if pretrained else "",
        last_stride=config.DISTILLER.STUDENT_LAST_STRIDE,
        num_classes=num_classes,
    )
    if teacher is None:
        return distiller_dict["NONE"](student, config)
    return distiller_dict[config.DISTILLER.TYPE](student, teacher, config)
