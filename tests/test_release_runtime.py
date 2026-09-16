"""Regression tests for portable configs, checkpoint round-trips and evaluation."""

import os
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AIR_Distiller"))

from config import cfg
from dataloader.make_dataloader import DataLoaderFactory
from dataloader.datasets.InShop import InShop
from distillers import distiller_dict
from models import model_dict
from processor import trainer_dict
from processor.inferencer import inference
from processor.trainer import KDTrainer
from tools.common import build_distiller, configure_device, load_checkpoint, load_config, load_teacher_checkpoint
from test_trikd import ToyModel, make_distiller


class ReleaseRuntimeTests(unittest.TestCase):
    def test_saved_effective_config_can_be_loaded_with_gpu_overrides(self):
        config = cfg.clone()
        config.EXPERIMENT.DEVICE_ID = "0"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(config.dump())
            restored = load_config(SimpleNamespace(cfg=str(path), opts=[]))
            self.assertEqual(restored.dump(), config.dump())
            restored = load_config(SimpleNamespace(cfg=str(path), opts=["EXPERIMENT.DEVICE_ID", "2"]))
            self.assertEqual(restored.EXPERIMENT.DEVICE_ID, "2")

    def test_all_public_configs_merge_and_register(self):
        paths = list((ROOT / "Training_Configs").rglob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=str(path.relative_to(ROOT))):
                config = cfg.clone()
                config.merge_from_file(str(path))
                self.assertIn(config.DISTILLER.TYPE, distiller_dict)
                self.assertIn(config.DISTILLER.STUDENT_NAME, model_dict)
                self.assertIn(config.DISTILLER.TEACHER_NAME, model_dict)
                self.assertIn(config.SOLVER.TRAINER, trainer_dict)
                for value in (config.DATASETS.ROOT_DIR, config.OUTPUT_DIR.ROOT_PATH,
                              config.DISTILLER.TEACHER_MODEL_PATH,
                              config.DISTILLER.STUDENT_PRETRAIN_PATH):
                    self.assertFalse(Path(value).is_absolute())

    def test_teacher_checkpoint_formats_round_trip(self):
        teacher = torch.nn.Linear(3, 2)
        state = teacher.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "teacher.pth"
            for prefix, wrapped in (("", False), ("student.", False), ("module.student.", True)):
                with self.subTest(prefix=prefix):
                    checkpoint = {prefix + key: value for key, value in state.items()}
                    torch.save({"state_dict": checkpoint} if wrapped else checkpoint, path)
                    target = torch.nn.Linear(3, 2)
                    load_teacher_checkpoint(target, path)
                    for key, value in target.state_dict().items():
                        torch.testing.assert_close(value, state[key])

    def test_evaluation_does_not_initialize_pretrained_models(self):
        config = cfg.clone()
        config.DISTILLER.TYPE = "TriKD"
        config.DISTILLER.TEACHER_MODEL_PATH = "missing-teacher.pth"
        config.DISTILLER.STUDENT_PRETRAIN_PATH = "missing-imagenet.pth"
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return ToyModel()

        with patch.dict(model_dict, {"ResNet18": factory, "ResNet101": factory}):
            build_distiller(config, num_classes=4, training=False)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(not call["pretrained"] for call in calls))
        self.assertTrue(all(not call.get("pretrained_path") for call in calls))

    def test_training_rejects_missing_teacher(self):
        config = cfg.clone()
        config.DISTILLER.TYPE = "TriKD"
        config.DISTILLER.TEACHER_MODEL_PATH = "missing-teacher.pth"
        with self.assertRaisesRegex(FileNotFoundError, "Teacher checkpoint"):
            build_distiller(config, num_classes=4, training=True)

    def test_full_checkpoint_and_raw_student_export_round_trip(self):
        model, config = make_distiller("directional")
        with tempfile.TemporaryDirectory() as folder:
            config.OUTPUT_DIR.ROOT_PATH = folder
            config.OUTPUT_DIR.EXPERIMENT_NAME = ""
            trainer = KDTrainer.__new__(KDTrainer)
            trainer.cfg = config
            trainer.distiller = torch.nn.DataParallel(model)
            trainer.save_checkpoint(1)
            restored, _ = make_distiller("directional")
            restored.load_state_dict(load_checkpoint(Path(folder) / f"{config.DISTILLER.TYPE}_1.pth"))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value.cpu())
            student = ToyModel()
            student.load_state_dict(load_checkpoint(Path(folder) / "student_1.pth"))
            for key, value in model.student.state_dict().items():
                torch.testing.assert_close(student.state_dict()[key], value.cpu())

    def test_flip_evaluation_accepts_small_feature_dimensions_and_parallel_wrapper(self):
        class RetrievalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward_query(self, image):
                return image.mean((-2, -1)) * self.scale

            forward_gallery = forward_query

        images = torch.eye(3)[:, :, None, None].expand(-1, -1, 2, 2)
        query = [(images[:2], torch.tensor([0, 1]), torch.tensor([0, 0]))]
        gallery = [(images, torch.tensor([0, 1, 2]), torch.tensor([1, 1, 1]))]
        config = cfg.clone()
        config.TEST.FLIP_FEATS = "on"
        with tempfile.TemporaryDirectory() as folder:
            config.OUTPUT_DIR.ROOT_PATH = folder
            config.OUTPUT_DIR.EXPERIMENT_NAME = ""
            cmc, mean_ap, mean_inp = inference(
                config, torch.nn.DataParallel(RetrievalModel()), query, gallery,
            )
            self.assertEqual(mean_ap, 1.0)
            self.assertEqual(mean_inp, 1.0)
            self.assertEqual(cmc[0], 1.0)
            self.assertIn("mAP: 100.00%", (Path(folder) / "test_acc.txt").read_text())

    def test_evaluation_does_not_construct_a_training_sampler(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta").mkdir()
            entries = []
            for category in ("apple_pie", "bread_pudding"):
                (root / "images" / category).mkdir(parents=True)
                for index in range(3):
                    Image.new("RGB", (16, 16)).save(root / "images" / category / f"{index}.jpg")
                    entries.append(f"{category}/{index}")
            (root / "meta/train.txt").write_text("\n".join(entries))
            (root / "meta/test.txt").write_text("\n".join(entries))
            config = cfg.clone()
            config.DATASETS.NAMES = "Food101"
            config.DATASETS.ROOT_DIR = folder
            config.DATALOADER.NUM_WORKERS = 0
            config.SOLVER.IMS_PER_BATCH = 13  # Invalid for the training triplet sampler.
            query, gallery, classes = DataLoaderFactory(config).create_evaluation_dataloaders()
            self.assertEqual((len(query.dataset), len(gallery.dataset), classes), (2, 4, 2))
            self.assertEqual(next(iter(query))[0].shape[1:], (3, 64, 64))

    def test_filename_ids_ignore_digits_in_parent_directories(self):
        with tempfile.TemporaryDirectory(prefix="trikd_d3_data_") as folder:
            for split in ("train", "query", "gallery"):
                directory = Path(folder) / "InShop" / split
                directory.mkdir(parents=True)
                (directory / "17_sample.jpg").touch()
            data = InShop(root=folder, verbose=False)
            self.assertEqual(data.query[0][1], 16)

    def test_shell_cuda_visibility_is_preserved(self):
        config = cfg.clone()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                patch("torch.cuda.is_available", return_value=True):
            configure_device(config, training=True)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")

    def test_cli_help_works_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            for entry in ("train.py", "test.py"):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "AIR_Distiller/tools" / entry), "--help"],
                    cwd=folder, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--cfg", result.stdout)


if __name__ == "__main__":
    unittest.main()
