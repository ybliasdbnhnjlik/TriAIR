import copy
import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AIR_Distiller"))

from config import cfg
from distillers.TriKD import TriKD, degradation_direction_distance, teacher_classifier_lsd_loss
from models.utils.class_block import ClassBlock
from processor.trainer import BaseTrainer, KDTrainer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))
        self.local = nn.Conv2d(8, 256, 1)
        self.fc = ClassBlock(8, 4, num_bottleneck=512)

    def forward(self, image):
        base = self.base(F.adaptive_avg_pool2d(image, 4))
        pooled = base.mean((2, 3))
        logits, retrieval = self.fc(pooled)
        return logits, {
            "feats": [base, base, self.local(base), base],
            "pooled_feat": pooled,
            "retrieval_feat": retrieval,
        }


def make_distiller(mode="scalar"):
    config = cfg.clone()
    config.EXPERIMENT.CUDA_AMP = False
    config.DISTILLER.TYPE = "TriKD"
    config.DISTILLER.STUDENT_NAME = "ResNet18"
    config.UGD.RA_ENABLED = True
    config.UGD.RA_MODE = mode
    config.UGD.RA_USE_CLASS_MASK = False
    config.D3.TOPK = 4
    return TriKD(ToyModel(), ToyModel(), config).train(), config


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_lsd_equal_features_and_small_batches(self):
        for batch in (1, 2, 8):
            with self.subTest(batch=batch):
                student = torch.randn(batch, 16, requires_grad=True)
                teacher = student.detach().clone().requires_grad_()
                classifier = nn.Linear(16, 7)
                loss = teacher_classifier_lsd_loss(student, teacher, classifier)
                self.assertLess(abs(loss.item()), 1e-6)
                loss.backward()
                self.assertTrue(torch.isfinite(student.grad).all())
                self.assertIsNone(teacher.grad)
                self.assertTrue(all(p.grad is None for p in classifier.parameters()))

    def _check_amp(self, device, dtype):
        student = torch.randn(24, 32, device=device, dtype=dtype, requires_grad=True)
        teacher = torch.randn_like(student).requires_grad_()
        classifier = nn.Linear(32, 13).to(device)
        reference = teacher_classifier_lsd_loss(student, teacher, classifier)
        reference_grad = torch.autograd.grad(reference, student)[0]
        with torch.autocast(device, dtype=dtype):
            actual = teacher_classifier_lsd_loss(student, teacher, classifier)
        actual_grad = torch.autograd.grad(actual, student)[0]
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-5)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))

    def test_lsd_cpu_autocast(self):
        self._check_amp("cpu", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_lsd_cuda_autocast(self):
        self._check_amp("cuda", torch.float16)

    def test_lsd_constant_logits_and_single_class_are_finite(self):
        for classes in (1, 7):
            with self.subTest(classes=classes):
                student = torch.zeros(8, 16, requires_grad=True)
                teacher = torch.zeros_like(student)
                classifier = nn.Linear(16, classes, bias=False)
                loss = teacher_classifier_lsd_loss(student, teacher, classifier)
                loss.backward()
                self.assertLess(abs(loss.item()), 1e-6)
                self.assertTrue(torch.isfinite(student.grad).all())

    def test_lsd_matches_shared_classifier_reference(self):
        student, teacher = torch.randn(6, 16), torch.randn(6, 16)
        classifier = nn.Linear(16, 9)
        def standardize(scores):
            return (scores - scores.mean(1, keepdim=True)) / (
                scores.var(1, unbiased=False, keepdim=True) + 1e-6
            ).sqrt()
        sz, tz = standardize(classifier(student)), standardize(classifier(teacher))
        expected = 4 * F.kl_div(
            F.log_softmax(sz / 2, dim=1), F.softmax(tz / 2, dim=1),
            reduction="batchmean",
        )
        torch.testing.assert_close(
            teacher_classifier_lsd_loss(student, teacher, classifier), expected
        )

    def test_lsd_single_query_is_supervised_and_batch_independent(self):
        student = torch.randn(6, 16, requires_grad=True)
        teacher = torch.randn_like(student)
        classifier = nn.Linear(16, 9, bias=False)
        loss = teacher_classifier_lsd_loss(student, teacher, classifier)
        single_losses = torch.stack([
            teacher_classifier_lsd_loss(student[i:i+1], teacher[i:i+1], classifier)
            for i in range(len(student))
        ])
        self.assertTrue((single_losses > 0).all())
        torch.testing.assert_close(loss, single_losses.mean())
        loss.backward()
        self.assertGreater(student.grad.norm().item(), 0.0)
        self.assertIsNone(classifier.weight.grad)

    def test_lsd_uses_only_teacher_classifier_and_preserves_student_ce(self):
        model, _ = make_distiller()
        logits, features = model.student(torch.randn(4, 3, 8, 8))
        teacher_features = torch.randn(4, 512, requires_grad=True)
        teacher_before = copy.deepcopy(model.teacher.state_dict())
        # The full teacher FC expects 8-D backbone features, not 512-D retrieval
        # features. Calling it here would also introduce unwanted BN handling.
        with patch.object(model.teacher.fc, "forward", side_effect=AssertionError):
            loss = model.semantic_lsd_loss(teacher_features, features["retrieval_feat"])
        loss.backward(retain_graph=True)
        self.assertGreater(model.student.fc.add_block[2].weight.grad.norm().item(), 0.0)
        self.assertGreater(model.student.base[0].weight.grad.norm().item(), 0.0)
        self.assertIsNone(model.student.fc.classifier.weight.grad)
        self.assertIsNone(teacher_features.grad)
        self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.teacher.parameters()))
        for name, value in model.teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
        model.student.zero_grad(set_to_none=True)
        model.ce_loss(logits, torch.tensor([0, 1, 2, 3])).backward()
        self.assertGreater(model.student.fc.classifier.weight.grad.norm().item(), 0.0)

    def test_lsd_config_values_override_legacy_values(self):
        config = cfg.clone()
        config.DISTILLER.STUDENT_NAME = "ResNet18"
        config.UGD.R_LSD_WEIGHT = 0.25
        config.UGD.R_LSD_TAU = 3.0
        config.UGD.R_LSD_TOPK = 1  # No candidate selection in the new LSD.
        model = TriKD(ToyModel(), ToyModel(), config)
        self.assertEqual((model.lsd_weight, model.lsd_tau), (0.25, 3.0))
        config.UGD.LSD_WEIGHT = 0.1
        config.UGD.LSD_TAU = 2.0
        model = TriKD(ToyModel(), ToyModel(), config)
        self.assertEqual((model.lsd_weight, model.lsd_tau), (0.1, 2.0))

    def test_bn_buffers_modes_and_gradients(self):
        module = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).train()
        module[1].eval()
        before = copy.deepcopy(module.state_dict())
        image = torch.randn(8, 4, requires_grad=True)
        result = TriKD._forward_with_batch_stats_no_update(module, image)
        result.square().mean().backward()
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(module[1].training)
        self.assertTrue(module[1].track_running_stats)
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_bn_modes_restored_on_exception(self):
        module = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(5, 2)).eval()
        with self.assertRaises(RuntimeError):
            TriKD._forward_with_batch_stats_no_update(module, torch.randn(8, 4))
        self.assertFalse(module[0].training)
        self.assertTrue(module[0].track_running_stats)

    def test_weight_normalization_and_zero_fallback(self):
        valid = torch.tensor([True, True, False])
        raw = torch.tensor([0.2, 0.8, 0.99])
        result = TriKD._budget_normalized_weight(raw, valid)
        self.assertAlmostEqual(result[valid].mean().item(), 1.0)
        self.assertGreater(result[1], result[0])
        for weights, mask in ((torch.zeros(3), valid), (raw, ~torch.ones(3).bool())):
            torch.testing.assert_close(
                TriKD._budget_normalized_weight(weights, mask), torch.ones(3)
            )

    def test_antialias_switch(self):
        model, _ = make_distiller()
        image = torch.randn(2, 3, 32, 32)
        for enabled in (False, True):
            model.ra_antialias = enabled
            expected = F.interpolate(
                image, size=(8, 8), mode="bilinear", align_corners=False,
                antialias=enabled,
            )
            torch.testing.assert_close(model._aligned_student_view(image, (8, 8)), expected)

    def test_forward_backward_and_diagnostics(self):
        for mode, space in (("scalar", "backbone"), ("scalar", "retrieval"), ("directional", "backbone")):
            with self.subTest(mode=mode, space=space):
                model, _ = make_distiller(mode)
                model.ra_accessibility_space = space
                teacher_before = copy.deepcopy(model.teacher.state_dict())
                image, kd_image = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
                teacher_image, target = torch.randn(4, 3, 32, 32), torch.tensor([0, 0, 1, 1])
                with patch.object(model, "semantic_lsd_loss", wraps=model.semantic_lsd_loss) as lsd:
                    _, losses = model(
                        image=image, kd_student_image=kd_image, kd_teacher_image=teacher_image,
                        target=target, kd_target=target,
                    )
                lsd.assert_called_once()
                teacher_features, student_features = lsd.call_args.args
                torch.testing.assert_close(
                    losses["loss_LS_KD"], model.lsd_weight * teacher_classifier_lsd_loss(
                        student_features, teacher_features, model.teacher.fc.classifier,
                        tau=model.lsd_tau,
                    ),
                )
                expected = sum(losses[k] for k in ("loss_ce", "loss_triplet", "loss_kd", "loss_LS_KD"))
                objective = BaseTrainer._compute_optimization_loss(losses)
                torch.testing.assert_close(objective, expected)
                torch.testing.assert_close(
                    losses["loss_kd_ugd"], model.kd_ugd_weight * (
                        model.alpha_ugd * losses["ugd_global"]
                        + model.beta_ugd * (losses["ugd_fine"] + losses["ugd_coarse"])
                    ),
                )
                objective.backward()
                self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.student.parameters() if p.grad is not None))
                self.assertEqual(model.student.base[1].num_batches_tracked.item(), 2)
                self.assertEqual(model.teacher.base[1].num_batches_tracked.item(), 0)
                self.assertEqual(model.projector[0].num_batches_tracked.item(), 1)
                self.assertGreater(model.projector[1].weight.grad.norm().item(), 0.0)
                self.assertGreater(model.projector_half[1].weight.grad.norm().item(), 0.0)
                for name, value in model.teacher.state_dict().items():
                    torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
                if mode == "directional":
                    self.assertNotIn("ra_accessibility", losses)
                    self.assertFalse(losses["ra_direction_fine_ratio"].requires_grad)
                    self.assertLessEqual(losses["ra_fine_loss_delta"].item(), 1e-6)
                    self.assertLessEqual(losses["ra_coarse_loss_delta"].item(), 1e-6)

    def test_blend_zero_and_empty_semantic_mask(self):
        model, _ = make_distiller()
        model.ra_blend = 0
        _, student = model.student(torch.randn(4, 3, 8, 8))
        _, aligned = model.student(torch.randn(4, 3, 8, 8))
        _, teacher = model.teacher(torch.randn(4, 3, 32, 32))
        _, teacher_low = model.teacher(torch.randn(4, 3, 32, 32))
        target = torch.zeros(4, dtype=torch.long)
        _, diag = model.ugd_loss(teacher, student, target, teacher_low, aligned)
        self.assertAlmostEqual(diag["ra_fine_loss_delta"].item(), 0.0)
        self.assertAlmostEqual(diag["ra_coarse_loss_delta"].item(), 0.0)
        model.ra_use_class_mask = True
        loss, diag = model.ugd_loss(teacher, student, target + 100, teacher_low, aligned)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(diag["ugd_fine"].item(), 0.0)
        self.assertEqual(diag["ugd_coarse"].item(), 0.0)

    def test_ra_disabled_needs_no_auxiliary_features(self):
        model, _ = make_distiller()
        model.ra_enabled = False
        _, student = model.student(torch.randn(4, 3, 8, 8))
        _, teacher = model.teacher(torch.randn(4, 3, 32, 32))
        loss, diag = model.ugd_loss(teacher, student, torch.zeros(4, dtype=torch.long))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(diag["ra_fine_loss_delta"].item(), 0.0)

    def test_kd_trainer_logs_details_without_double_counting(self):
        self._check_trainer("scalar")

    def test_directional_trainer_logs_details_without_double_counting(self):
        self._check_trainer("directional")

    def _check_trainer(self, mode):
        model, config = make_distiller(mode)
        config.SOLVER.LR_DECAY_TYPE = "WarmupCosineAnnealingLR"
        target = torch.tensor([0, 0, 1, 1])
        main_batch = (torch.randn(4, 3, 8, 8), target)
        kd_batch = (torch.randn(4, 3, 8, 8), torch.randn(4, 3, 32, 32), target)
        # The test intentionally exercises CPU training even on multi-GPU hosts.
        with patch("torch.cuda.device_count", return_value=0):
            trainer = KDTrainer(config, model, [main_batch, main_batch], [kd_batch], [], [])
        trainer.device = "cpu"
        with self.assertLogs("Asymmetric_Image_Retrieval.train", level=logging.INFO) as logs:
            trainer.train_epoch(1)
        text = "\n".join(logs.output)
        labels = ["UGD_Global:", "RA_FineDelta:"]
        if mode == "scalar":
            labels += ["RA_FineStd:", "RA_CoarseStd:"]
            self.assertEqual(trainer.ra_fine_weight_std_meter.count, 8)
        else:
            labels += ["RA_DirFineNorm:", "RA_DirFineRatio:", "RA_DirCoarseRatio:"]
            labels += ["RA_DirFineValid:", "RA_DirCoarseCount:", "RA_DirFineEmptyBatch:"]
            self.assertNotIn("RA_FineW:", text)
            self.assertNotIn("RA_Access:", text)
            grid = model.downsampling_pooling.pooling.output_size
            region_count = 8 * grid[0] * grid[1]
            self.assertEqual(trainer.ugd_detail_meters["ra_direction_fine_ratio"].count, region_count)
            self.assertEqual(trainer.ra_direction_valid_meters["fine"].sum, region_count)
            self.assertEqual(trainer.ra_direction_empty_meters["fine"].avg, 0)
        for label in labels:
            self.assertIn(label, text)
        self.assertEqual(trainer.ugd_detail_meters["ugd_global"].count, 8)
        self.assertAlmostEqual(
            float(trainer.loss_meter.avg),
            float(trainer.ce_meter.avg + trainer.tri_meter.avg + trainer.kd_meter.avg + trainer.ls_kd_meter.avg),
            places=5,
        )
        trainer.reset_meters()
        self.assertEqual(trainer._format_ugd_details(), "")

    def _make_directional_statistics_trainer(self):
        model, config = make_distiller("directional")
        config.SOLVER.LR_DECAY_TYPE = "WarmupCosineAnnealingLR"
        with patch("torch.cuda.device_count", return_value=0):
            return KDTrainer(config, model, [], [], [], [])

    @staticmethod
    def _empty_directional_statistics():
        return {
            "ra_direction_fine_valid_count": torch.tensor(0),
            "ra_direction_coarse_valid_count": torch.tensor(0),
            "ra_direction_fine_region_count": torch.tensor(32),
            "ra_direction_coarse_region_count": torch.tensor(8),
            "ra_direction_valid_ratio": torch.tensor(0.),
            "ra_direction_hr_lr_similarity": torch.tensor(0.5),
            "ra_direction_fine_norm": torch.tensor(0.),
            "ra_direction_coarse_norm": torch.tensor(0.),
            "ra_direction_fine_ratio": torch.tensor(0.),
            "ra_direction_coarse_ratio": torch.tensor(0.),
        }

    def test_directional_empty_epoch_reports_na_and_preserves_loss(self):
        trainer = self._make_directional_statistics_trainer()
        diagnostics = self._empty_directional_statistics()
        trainer._update_ugd_details(diagnostics, 2)
        text = trainer._format_ugd_details()
        for label in ("RA_DirFineNorm", "RA_DirCoarseNorm", "RA_DirFineRatio", "RA_DirCoarseRatio"):
            self.assertIn(f"{label}: N/A", text)
        self.assertIn("RA_DirFineCount: 0/32", text)
        self.assertIn("RA_DirCoarseValid: 0.00000", text)
        self.assertIn("RA_DirCoarseEmptyBatch: 1.00000", text)
        objective = torch.tensor(3., requires_grad=True)
        actual = BaseTrainer._compute_optimization_loss({"loss_kd": objective, **diagnostics})
        torch.testing.assert_close(actual, objective)
        actual.backward()
        self.assertEqual(objective.grad.item(), 1)
        trainer.reset_meters()
        self.assertEqual(trainer._format_ugd_details(), "")
        self.assertEqual(trainer.ra_direction_empty_meters["fine"].count, 0)

    def test_directional_statistics_weight_regions_across_batches_and_replicas(self):
        trainer = self._make_directional_statistics_trainer()
        # One empty batch followed by an uneven two-replica batch. In the
        # latter, only one replica has valid coarse regions.
        trainer._update_ugd_details(self._empty_directional_statistics(), 2)
        diagnostics = {
            "ra_direction_fine_valid_count": torch.tensor([1, 3]),
            "ra_direction_coarse_valid_count": torch.tensor([0, 2]),
            "ra_direction_fine_region_count": torch.tensor([16, 32]),
            "ra_direction_coarse_region_count": torch.tensor([4, 8]),
            "ra_direction_valid_ratio": torch.tensor([1 / 20, 5 / 40]),
            "ra_direction_hr_lr_similarity": torch.tensor([0.6, 0.9]),
            "ra_direction_fine_norm": torch.tensor([2., 4.]),
            "ra_direction_coarse_norm": torch.tensor([0., 6.]),
            "ra_direction_fine_ratio": torch.tensor([0.9, 1.]),
            "ra_direction_coarse_ratio": torch.tensor([0., 0.95]),
            "ugd_fine": torch.tensor([1., 3.]),
        }
        trainer._update_ugd_details(diagnostics, 3)
        meters = trainer.ugd_detail_meters
        self.assertAlmostEqual(meters["ra_direction_fine_norm"].avg, 3.5)
        self.assertAlmostEqual(meters["ra_direction_fine_ratio"].avg, 0.975)
        self.assertEqual(meters["ra_direction_fine_ratio"].count, 4)
        self.assertAlmostEqual(meters["ra_direction_coarse_norm"].avg, 6.)
        self.assertAlmostEqual(meters["ra_direction_coarse_ratio"].avg, 0.95)
        self.assertEqual(meters["ra_direction_coarse_ratio"].count, 2)
        self.assertAlmostEqual(meters["ra_direction_valid_ratio"].avg, 6 / 100)
        self.assertAlmostEqual(meters["ra_direction_hr_lr_similarity"].avg, 0.68)
        self.assertAlmostEqual(trainer.ra_direction_valid_meters["fine"].avg, 4 / 80)
        self.assertAlmostEqual(trainer.ra_direction_valid_meters["coarse"].avg, 2 / 20)
        for scale in ("fine", "coarse"):
            self.assertEqual(trainer.ra_direction_empty_meters[scale].avg, 0.5)
        # Training-loss logging still uses its previous image/batch weighting.
        self.assertEqual(meters["ugd_fine"].avg, 2.)
        self.assertEqual(meters["ugd_fine"].count, 3)
        self.assertNotIn("N/A", trainer._format_ugd_details())
        objective = torch.tensor(2., requires_grad=True)
        torch.testing.assert_close(
            BaseTrainer._compute_optimization_loss({"loss_kd": objective, **diagnostics}),
            objective,
        )

    def test_directional_uses_retrieval_targets_not_scalar_gate(self):
        model, _ = make_distiller("directional")
        _, student = model.student(torch.randn(4, 3, 8, 8))
        _, aligned = model.student(torch.randn(4, 3, 8, 8))
        _, teacher = model.teacher(torch.randn(4, 3, 32, 32))
        _, low = model.teacher(torch.randn(4, 3, 32, 32))
        target = torch.zeros(4, dtype=torch.long)
        # Directional mode must not enter the old gate, even though a saved
        # RA_ACCESSIBILITY_SPACE value can still be "backbone".
        with patch.object(model, "_resolution_accessibility", side_effect=AssertionError), \
             patch.object(model, "_budget_normalized_weight", side_effect=AssertionError), \
             patch.object(model.teacher.fc, "forward", wraps=model.teacher.fc.forward) as fc:
            _, diag = model.ugd_loss(teacher, student, target, low, aligned)
        self.assertEqual(fc.call_count, 4)  # high/low x fine/coarse
        with torch.no_grad():
            def local_features(pool, features):
                return F.normalize(model.teacher.fc(pool(features["feats"][-1]))[1].float(), dim=1)
            high_fine = local_features(model.downsampling_pooling, teacher)
            low_fine = local_features(model.downsampling_pooling, low)
            student_fine = F.normalize(model.projector(model.downsampling_pooling(aligned["feats"][2])).float(), dim=1)
            expected = degradation_direction_distance(student_fine, high_fine, low_fine).mean()
        torch.testing.assert_close(diag["ugd_fine"], expected)
        self.assertGreater(diag["ra_direction_fine_norm"].item(), 0.0)

    def test_directional_zero_rho_and_empty_mask(self):
        model, _ = make_distiller("directional")
        model.ra_direction_rho = 0
        _, student = model.student(torch.randn(4, 3, 8, 8))
        _, aligned = model.student(torch.randn(4, 3, 8, 8))
        _, teacher = model.teacher(torch.randn(4, 3, 32, 32))
        _, low = model.teacher(torch.randn(4, 3, 32, 32))
        target = torch.zeros(4, dtype=torch.long)
        _, diag = model.ugd_loss(teacher, student, target, low, aligned)
        for scale in ("fine", "coarse"):
            self.assertAlmostEqual(diag[f"ra_{scale}_loss_delta"].item(), 0.0)
            self.assertAlmostEqual(diag[f"ra_direction_{scale}_ratio"].item(), 1.0)
        # With no semantic regions, local losses and gradients are zero.
        model.ra_direction_rho = 0.25
        model.ra_use_class_mask = True
        loss, diag = model.ugd_loss(teacher, student, target + 100, low, aligned)
        loss.backward()
        self.assertTrue(all(torch.isfinite(v).all() for v in diag.values()))
        for scale in ("fine", "coarse"):
            self.assertEqual(diag[f"ugd_{scale}"].item(), 0.0)
            self.assertEqual(diag[f"ra_direction_{scale}_valid_count"].item(), 0)
            self.assertGreater(diag[f"ra_direction_{scale}_region_count"].item(), 0)
        self.assertEqual(model.projector[1].weight.grad.norm().item(), 0.0)
        # Disabling RA still works without an auxiliary view in either mode.
        model.ra_enabled = False
        loss, _ = model.ugd_loss(teacher, student, target)
        self.assertTrue(torch.isfinite(loss))

    def test_directional_config_validation_and_legacy_default(self):
        config = cfg.clone()
        config.DISTILLER.STUDENT_NAME = "ResNet18"
        self.assertEqual(TriKD(ToyModel(), ToyModel(), config).ra_mode, "scalar")
        config.UGD.RA_MODE = "directional"
        for rho in (-0.1, 1.0, float("nan")):
            config.UGD.RA_DIRECTION_RHO = rho
            with self.assertRaises(ValueError):
                TriKD(ToyModel(), ToyModel(), config)

    def test_food172_configs(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root / "Training_Configs" / "Food172").glob("*/TriKD.yaml"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            config = cfg.clone()
            config.merge_from_file(str(path))
            self.assertTrue(config.UGD.RA_ANTIALIAS)
            self.assertEqual(config.UGD.RA_MODE, "directional")
            self.assertEqual(config.UGD.RA_DIRECTION_RHO, 0.25)
            self.assertIn("Directional", config.OUTPUT_DIR.EXPERIMENT_NAME)
            self.assertTrue(config.OUTPUT_DIR.EXPERIMENT_NAME.endswith("TeacherClassifierLSD_seed2024"))
            self.assertEqual(config.SOLVER.SEED, 2024)
            self.assertEqual(config.UGD.LSD_WEIGHT, 0.1)
            self.assertEqual(config.UGD.LSD_TAU, 2.0)
            self.assertEqual(config.D3.TOPK, 10)


class DirectionalDistanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_formula_gradient_and_bounds(self):
        high = F.normalize(torch.randn(32, 16), dim=1).requires_grad_()
        low = F.normalize(torch.randn(32, 16), dim=1).requires_grad_()
        student = F.normalize(torch.randn(32, 16), dim=1).requires_grad_()
        rho = 0.25
        delta = high.detach() - low.detach()
        error = student - high.detach()
        original = error.norm(dim=1)
        expected = (error.square().sum(1) - rho * (error * delta).sum(1).square()
                    / (delta.square().sum(1) + 1e-6)).sqrt()
        actual = degradation_direction_distance(student, high, low, rho)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), student, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), student)[0],
        )
        self.assertTrue((actual <= original + 1e-6).all())
        self.assertTrue((actual >= (1 - rho) ** 0.5 * original - 1e-6).all())
        actual.sum().backward()
        self.assertIsNone(high.grad)
        self.assertIsNone(low.grad)

    def test_parallel_orthogonal_and_gradient_direction(self):
        high = torch.zeros(3, 2)
        low = torch.tensor([[-1., 0.]]).repeat(3, 1)
        student = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
        rho = 0.5
        coeff = 1 - rho / (1 + 1e-6)
        distance = degradation_direction_distance(student, high, low, rho)
        torch.testing.assert_close(distance, torch.tensor([coeff ** 0.5, 1., (coeff + 1) ** 0.5]))
        distance.sum().backward()
        self.assertLess(student.grad[2, 0].item(), student.grad[2, 1].item())

    def test_zero_error_zero_degradation_and_zero_rho(self):
        for equal_teacher in (False, True):
            for equal_student in (False, True):
                for rho in (0., 0.25, 0.999):
                    with self.subTest(equal_teacher=equal_teacher, equal_student=equal_student, rho=rho):
                        high = torch.randn(4, 8)
                        low = high.clone() if equal_teacher else torch.randn_like(high)
                        student = (high.clone() if equal_student else torch.randn_like(high)).requires_grad_()
                        distance = degradation_direction_distance(student, high, low, rho)
                        distance.sum().backward()
                        self.assertTrue(torch.isfinite(student.grad).all())
                        if equal_teacher or equal_student or rho == 0:
                            torch.testing.assert_close(distance, (student - high).norm(dim=1))
                        if equal_student:
                            torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_tiny_degradation_is_finite(self):
        high = torch.zeros(4, 8)
        low = torch.full_like(high, 1e-10)
        student = torch.randn_like(high, requires_grad=True)
        distance = degradation_direction_distance(student, high, low)
        torch.testing.assert_close(distance, student.norm(dim=1))
        distance.sum().backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def _check_amp(self, device, dtype):
        student = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=True)
        high, low = torch.randn_like(student), torch.randn_like(student)
        expected = degradation_direction_distance(student, high, low)
        with torch.autocast(device, dtype=dtype):
            actual = degradation_direction_distance(student, high, low)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_cpu_autocast(self):
        self._check_amp("cpu", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_autocast(self):
        self._check_amp("cuda", torch.float16)


if __name__ == "__main__":
    unittest.main()
