import os
import time
import logging
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from utils.meter import AverageMeter
from solver import make_optimizer, WarmupMultiStepLR, WarmupCosineAnnealingLR
from .inferencer import inference


class BaseTrainer(object):
    _UGD_DETAIL_LABELS = {
        "ugd_global": "UGD_Global",
        "ugd_fine": "UGD_Fine",
        "ugd_coarse": "UGD_Coarse",
        "ra_fine_loss_delta": "RA_FineDelta",
        "ra_coarse_loss_delta": "RA_CoarseDelta",
        "ra_direction_valid_ratio": "RA_DirValid",
        "ra_direction_hr_lr_similarity": "RA_DirSim",
        "ra_direction_fine_norm": "RA_DirFineNorm",
        "ra_direction_coarse_norm": "RA_DirCoarseNorm",
        "ra_direction_fine_ratio": "RA_DirFineRatio",
        "ra_direction_coarse_ratio": "RA_DirCoarseRatio",
    }
    _RA_DIRECTION_CONDITIONAL_SCALES = {
        "ra_direction_fine_norm": "fine",
        "ra_direction_coarse_norm": "coarse",
        "ra_direction_fine_ratio": "fine",
        "ra_direction_coarse_ratio": "coarse",
    }
    _RA_DIRECTION_COUNT_KEYS = frozenset(
        f"ra_direction_{scale}_{kind}_count"
        for scale in ("fine", "coarse")
        for kind in ("valid", "region")
    )
    # These entries expose the components of an already aggregated loss for
    # logging only.  They must not be summed into the optimization objective a
    # second time.
    _DIAGNOSTIC_LOSS_KEYS = frozenset(
        {
            "loss_kd_ugd",
            "loss_kd_d3",
            "ra_accessibility",
            "ra_valid_ratio",
            "ra_hr_lr_similarity",
            "ra_fine_effective_weight",
            "ra_coarse_effective_weight",
            "ra_fine_weight_std",
            "ra_coarse_weight_std",
        }
    ).union(_UGD_DETAIL_LABELS, _RA_DIRECTION_COUNT_KEYS)

    def __init__(self, cfg, distiller, student_train_loader, distillation_loader, query_loader, gallery_loader):
        self.cfg = cfg
        self.distiller = distiller
        self.student_train_loader = student_train_loader
        self.distillation_loader = distillation_loader

        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_amp = cfg.EXPERIMENT.CUDA_AMP
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.max_epochs = cfg.SOLVER.MAX_EPOCHS

        self.scaler = GradScaler() if self.cuda_amp else None

        self.logger = logging.getLogger("Asymmetric_Image_Retrieval.train")

        self.optimizer = make_optimizer(cfg, distiller)

        if cfg.SOLVER.LR_DECAY_TYPE == 'WarmupMultiStepLR':
            self.scheduler = WarmupMultiStepLR(
                self.optimizer,
                cfg.SOLVER.LR_DECAY_STEPS,
                cfg.SOLVER.LR_DECAY_GAMMA,
                cfg.SOLVER.LR_WARMUP_FACTOR,
                cfg.SOLVER.LR_WARMUP_EPOCHS,
                cfg.SOLVER.LR_WARMUP_METHOD,
            )
            self.logger.info("use WarmupMultiStepLR, delay_step:{}".format(cfg.SOLVER.LR_DECAY_STEPS))

        elif cfg.SOLVER.LR_DECAY_TYPE == 'WarmupCosineAnnealingLR':
            self.scheduler = WarmupCosineAnnealingLR(
                self.optimizer,
                cfg.SOLVER.MAX_EPOCHS,
                cfg.SOLVER.LR_DECAY_STEPS[0],
                cfg.SOLVER.LR_DECAY_ETA_MIN_LR,
                cfg.SOLVER.LR_WARMUP_FACTOR,
                cfg.SOLVER.LR_WARMUP_EPOCHS,
                cfg.SOLVER.LR_WARMUP_METHOD,
            )
            self.logger.info("use WarmupCosineAnnealingLR, delay_step:{}".format(cfg.SOLVER.LR_DECAY_STEPS[0]))

        self.logger.info("Trainer initialized.")

        # Meters for tracking metrics
        self.loss_meter = AverageMeter()
        self.ce_meter = AverageMeter()
        self.tri_meter = AverageMeter()
        self.kd_meter = AverageMeter()
        # ===== 新增：更细粒度的 KD meter =====
        self.kd_ugd_meter = AverageMeter()   # loss_kd_ugd
        self.kd_d3_meter = AverageMeter()    # loss_kd_d3
        # ===== 新增：LS-KD 的 meter =====
        self.ls_kd_meter = AverageMeter()
        self.ra_accessibility_meter = AverageMeter()
        self.ra_valid_ratio_meter = AverageMeter()
        self.ra_similarity_meter = AverageMeter()
        self.ra_fine_weight_meter = AverageMeter()
        self.ra_coarse_weight_meter = AverageMeter()
        self.ra_fine_weight_std_meter = AverageMeter()
        self.ra_coarse_weight_std_meter = AverageMeter()
        self.ugd_detail_meters = {
            name: AverageMeter() for name in self._UGD_DETAIL_LABELS
        }
        self.ra_direction_valid_meters = {
            scale: AverageMeter() for scale in ("fine", "coarse")
        }
        self.ra_direction_empty_meters = {
            scale: AverageMeter() for scale in ("fine", "coarse")
        }
        self.acc_meter = AverageMeter()

    @classmethod
    def _compute_optimization_loss(cls, losses_dict):
        optimization_losses = [
            value.mean()
            for name, value in losses_dict.items()
            if name not in cls._DIAGNOSTIC_LOSS_KEYS
        ]
        if not optimization_losses:
            raise ValueError("losses_dict does not contain an optimization loss")
        return sum(optimization_losses)

    def _update_ugd_details(self, losses_dict, batch_size):
        valid_counts, region_counts = {}, {}
        total_valid, total_regions = 0, 0
        for scale in self.ra_direction_valid_meters:
            valid_key = f"ra_direction_{scale}_valid_count"
            if valid_key not in losses_dict:
                continue
            # Scalars on one GPU, vectors of replica counts under DataParallel.
            valid_counts[scale] = losses_dict[valid_key].detach().reshape(-1)
            region_counts[scale] = losses_dict[
                f"ra_direction_{scale}_region_count"
            ].detach().reshape(-1)
            valid = valid_counts[scale].sum().item()
            regions = region_counts[scale].sum().item()
            self.ra_direction_valid_meters[scale].update(valid / regions, regions)
            # An empty batch means no valid regions across the whole batch,
            # not just an empty DataParallel replica.
            self.ra_direction_empty_meters[scale].update(float(valid == 0))
            total_valid += valid
            total_regions += regions

        for name, meter in self.ugd_detail_meters.items():
            if name not in losses_dict:
                continue
            values = losses_dict[name].detach().reshape(-1)
            scale = self._RA_DIRECTION_CONDITIONAL_SCALES.get(name)
            if scale in valid_counts:
                count = valid_counts[scale].sum().item()
                if count:
                    value_sum = (values * valid_counts[scale]).sum().item()
                    meter.update(value_sum / count, count)
            elif name == "ra_direction_valid_ratio" and total_regions:
                meter.update(total_valid / total_regions, total_regions)
            elif name == "ra_direction_hr_lr_similarity" and total_regions:
                counts = sum(region_counts.values())
                meter.update((values * counts).sum().item() / total_regions, total_regions)
            else:
                # Actual training losses retain their existing batch averaging.
                meter.update(values.mean().item(), batch_size)

    def _format_ugd_details(self):
        parts = []
        for name, meter in self.ugd_detail_meters.items():
            if meter.count:
                parts.append(f"{self._UGD_DETAIL_LABELS[name]}: {meter.avg:.5f}, ")
            else:
                scale = self._RA_DIRECTION_CONDITIONAL_SCALES.get(name)
                if scale and self.ra_direction_valid_meters[scale].count:
                    parts.append(f"{self._UGD_DETAIL_LABELS[name]}: N/A, ")
        for scale, meter in self.ra_direction_valid_meters.items():
            if meter.count:
                label = scale.capitalize()
                parts.append(
                    f"RA_Dir{label}Valid: {meter.avg:.5f}, "
                    f"RA_Dir{label}Count: {meter.sum:.0f}/{meter.count:.0f}, "
                    f"RA_Dir{label}EmptyBatch: {self.ra_direction_empty_meters[scale].avg:.5f}, "
                )
        return "".join(parts)

    def _format_scalar_ra_details(self):
        # Directional RA has no scalar region weights. Do not log zero-valued
        # placeholders as if they described the new mechanism.
        if not self.ra_accessibility_meter.count:
            return ""
        return (
            f"RA_Access: {self.ra_accessibility_meter.avg:.3f}, "
            f"RA_Valid: {self.ra_valid_ratio_meter.avg:.3f}, "
            f"RA_Sim: {self.ra_similarity_meter.avg:.3f}, "
            f"RA_FineW: {self.ra_fine_weight_meter.avg:.3f}, "
            f"RA_CoarseW: {self.ra_coarse_weight_meter.avg:.3f}, "
            f"RA_FineStd: {self.ra_fine_weight_std_meter.avg:.5f}, "
            f"RA_CoarseStd: {self.ra_coarse_weight_std_meter.avg:.5f}, "
        )

    def reset_meters(self):
        self.loss_meter.reset()
        self.ce_meter.reset()
        self.tri_meter.reset()
        self.kd_meter.reset()
        # ===== 新增：重置各类 KD meter =====
        self.kd_ugd_meter.reset()
        self.kd_d3_meter.reset()
        self.ls_kd_meter.reset()
        self.ra_accessibility_meter.reset()
        self.ra_valid_ratio_meter.reset()
        self.ra_similarity_meter.reset()
        self.ra_fine_weight_meter.reset()
        self.ra_coarse_weight_meter.reset()
        self.ra_fine_weight_std_meter.reset()
        self.ra_coarse_weight_std_meter.reset()
        for meter in self.ugd_detail_meters.values():
            meter.reset()
        for meter in self.ra_direction_valid_meters.values():
            meter.reset()
        for meter in self.ra_direction_empty_meters.values():
            meter.reset()
        self.acc_meter.reset()

    def save_checkpoint(self, epoch):
        distillaer_save_path = os.path.join(
            self.cfg.OUTPUT_DIR.ROOT_PATH,
            self.cfg.OUTPUT_DIR.EXPERIMENT_NAME,
            f"{self.cfg.DISTILLER.TYPE}_{epoch}.pth",
        )

        model_state = (
            self.distiller.module.state_dict()
            if isinstance(self.distiller, torch.nn.DataParallel)
            else self.distiller.state_dict()
        )
        torch.save(model_state, distillaer_save_path)

    def train_epoch(self, epoch):
        self.reset_meters()
        self.distiller.train()

        start_time = time.time()
        for n_iter, (img, target) in enumerate(self.student_train_loader):
            self.optimizer.zero_grad()

            img, target = img.to(self.device), target.to(self.device)

            if self.cuda_amp:
                with autocast(device_type="cuda"):
                    preds, losses_dict = self.distiller(image=img, target=target)

                loss = self._compute_optimization_loss(losses_dict)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                preds, losses_dict = self.distiller(image=img, target=target)
                loss = self._compute_optimization_loss(losses_dict)
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_size = img.size(0)
            acc = (preds.max(1)[1] == target).float().mean().item()
            self.loss_meter.update(loss, batch_size)
            self.ce_meter.update(
                losses_dict["loss_ce"].mean().cpu().detach().numpy(), batch_size
            )
            self.tri_meter.update(
                losses_dict["loss_triplet"].mean().cpu().detach().numpy(), batch_size
            )
            self.kd_meter.update(
                losses_dict["loss_kd"].mean().cpu().detach().numpy(), batch_size
            )
            # ===== 新增：细分 KD：UGD / D3 / LS-KD =====
            if "loss_kd_ugd" in losses_dict:
                self.kd_ugd_meter.update(
                    losses_dict["loss_kd_ugd"].mean().cpu().detach().numpy(), batch_size
                )
            if "loss_kd_d3" in losses_dict:
                self.kd_d3_meter.update(
                    losses_dict["loss_kd_d3"].mean().cpu().detach().numpy(), batch_size
                )
            if "loss_LS_KD" in losses_dict:
                self.ls_kd_meter.update(
                    losses_dict["loss_LS_KD"].mean().cpu().detach().numpy(), batch_size
                )
            if "ra_accessibility" in losses_dict:
                self.ra_accessibility_meter.update(
                    losses_dict["ra_accessibility"].item(), batch_size
                )
                self.ra_valid_ratio_meter.update(
                    losses_dict["ra_valid_ratio"].item(), batch_size
                )
                self.ra_similarity_meter.update(
                    losses_dict["ra_hr_lr_similarity"].item(), batch_size
                )
                self.ra_fine_weight_meter.update(
                    losses_dict["ra_fine_effective_weight"].item(), batch_size
                )
                self.ra_coarse_weight_meter.update(
                    losses_dict["ra_coarse_effective_weight"].item(), batch_size
                )
                self.ra_fine_weight_std_meter.update(
                    losses_dict["ra_fine_weight_std"].item(), batch_size
                )
                self.ra_coarse_weight_std_meter.update(
                    losses_dict["ra_coarse_weight_std"].item(), batch_size
                )
            self.acc_meter.update(acc, 1)
            self._update_ugd_details(losses_dict, batch_size)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        self.logger.info(
            f"Epoch[{epoch}] Loss: {self.loss_meter.avg:.3f}, "
            f"CE: {self.ce_meter.avg:.3f}, "
            f"TRI: {self.tri_meter.avg:.3f}, "
            f"KD: {self.kd_meter.avg:.3f}, "
            f"KD_UGD: {self.kd_ugd_meter.avg:.3f}, "
            f"KD_D3: {self.kd_d3_meter.avg:.3f}, "
            f"LS_KD: {self.ls_kd_meter.avg:.3f}, "
            f"{self._format_scalar_ra_details()}"
            f"{self._format_ugd_details()}"
            f"Acc: {self.acc_meter.avg:.3f}, "
            f"Base LR: {self.scheduler.get_last_lr()[0]:.2e}, "
            f"Time/Batch: {time_per_batch:.3f}s"
        )

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(1, self.max_epochs + 1):
            self.train_epoch(epoch)
            self.scheduler.step()
            if epoch % self.checkpoint_period == 0 or epoch == self.max_epochs:
                self.save_checkpoint(epoch)
                inference(self.cfg, self.distiller, self.query_loader, self.gallery_loader)
        self.logger.info("Training completed.")


class KDTrainer(BaseTrainer):
    def save_checkpoint(self, epoch):
        distillaer_save_path = os.path.join(
            self.cfg.OUTPUT_DIR.ROOT_PATH,
            self.cfg.OUTPUT_DIR.EXPERIMENT_NAME,
            f"{self.cfg.DISTILLER.TYPE}_{epoch}.pth",
        )

        model_state = (
            self.distiller.module.state_dict()
            if isinstance(self.distiller, torch.nn.DataParallel)
            else self.distiller.state_dict()
        )
        torch.save(model_state, distillaer_save_path)

        student_model = {
            name.removeprefix("student."): parameter
            for name, parameter in model_state.items()
            if name.startswith("student.")
        }

        student_save_path = os.path.join(
            self.cfg.OUTPUT_DIR.ROOT_PATH,
            self.cfg.OUTPUT_DIR.EXPERIMENT_NAME,
            "student_{}.pth".format(epoch),
        )
        torch.save(student_model, student_save_path)

    def train_epoch(self, epoch):
        self.reset_meters()
        self.distiller.train()

        start_time = time.time()

        # The supervised loader defines an epoch.  ``zip`` used to truncate it
        # when the KD loader had fewer batches (e.g. 258 vs. 685 on Food172),
        # silently discarding most supervised updates.  Restart the KD iterator
        # if necessary instead.
        distillation_iterator = iter(self.distillation_loader)
        for n_iter, (img, target) in enumerate(self.student_train_loader):
            try:
                kd_student_img, kd_teacher_img, kd_target = next(
                    distillation_iterator
                )
            except StopIteration:
                distillation_iterator = iter(self.distillation_loader)
                try:
                    kd_student_img, kd_teacher_img, kd_target = next(
                        distillation_iterator
                    )
                except StopIteration as error:
                    raise ValueError("distillation_loader must not be empty") from error
            self.optimizer.zero_grad()

            img, target = img.to(self.device), target.to(self.device)
            kd_student_img, kd_teacher_img, kd_target = (
                kd_student_img.to(self.device),
                kd_teacher_img.to(self.device),
                kd_target.to(self.device),
            )

            if self.cuda_amp:
                with autocast(device_type="cuda"):
                    preds, losses_dict = self.distiller(
                        image=img,
                        kd_student_image=kd_student_img,
                        kd_teacher_image=kd_teacher_img,
                        target=target,
                        kd_target=kd_target,
                    )

                loss = self._compute_optimization_loss(losses_dict)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                preds, losses_dict = self.distiller(
                    image=img,
                    kd_student_image=kd_student_img,
                    kd_teacher_image=kd_teacher_img,
                    target=target,
                    kd_target=kd_target,
                )
                loss = self._compute_optimization_loss(losses_dict)
                loss.backward()
                self.optimizer.step()

            # Update metrics
            batch_size = img.size(0)
            acc = (preds.max(1)[1] == target).float().mean().item()
            self.loss_meter.update(loss, batch_size)
            self.ce_meter.update(
                losses_dict["loss_ce"].mean().cpu().detach().numpy(), batch_size
            )
            self.tri_meter.update(
                losses_dict["loss_triplet"].mean().cpu().detach().numpy(), batch_size
            )
            self.kd_meter.update(
                losses_dict["loss_kd"].mean().cpu().detach().numpy(), batch_size
            )
            # ===== 新增：细分 KD：UGD / D3 / LS-KD =====
            if "loss_kd_ugd" in losses_dict:
                self.kd_ugd_meter.update(
                    losses_dict["loss_kd_ugd"].mean().cpu().detach().numpy(), batch_size
                )
            if "loss_kd_d3" in losses_dict:
                self.kd_d3_meter.update(
                    losses_dict["loss_kd_d3"].mean().cpu().detach().numpy(), batch_size
                )
            if "loss_LS_KD" in losses_dict:
                self.ls_kd_meter.update(
                    losses_dict["loss_LS_KD"].mean().cpu().detach().numpy(), batch_size
                )
            if "ra_accessibility" in losses_dict:
                self.ra_accessibility_meter.update(
                    losses_dict["ra_accessibility"].item(), batch_size
                )
                self.ra_valid_ratio_meter.update(
                    losses_dict["ra_valid_ratio"].item(), batch_size
                )
                self.ra_similarity_meter.update(
                    losses_dict["ra_hr_lr_similarity"].item(), batch_size
                )
                self.ra_fine_weight_meter.update(
                    losses_dict["ra_fine_effective_weight"].item(), batch_size
                )
                self.ra_coarse_weight_meter.update(
                    losses_dict["ra_coarse_effective_weight"].item(), batch_size
                )
                self.ra_fine_weight_std_meter.update(
                    losses_dict["ra_fine_weight_std"].mean().item(), batch_size
                )
                self.ra_coarse_weight_std_meter.update(
                    losses_dict["ra_coarse_weight_std"].mean().item(), batch_size
                )
            self.acc_meter.update(acc, 1)
            self._update_ugd_details(losses_dict, kd_student_img.size(0))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        self.logger.info(
            f"Epoch[{epoch}] Loss: {self.loss_meter.avg:.3f}, "
            f"CE: {self.ce_meter.avg:.3f}, "
            f"TRI: {self.tri_meter.avg:.3f}, "
            f"KD: {self.kd_meter.avg:.3f}, "
            f"KD_UGD: {self.kd_ugd_meter.avg:.3f}, "
            f"KD_D3: {self.kd_d3_meter.avg:.3f}, "
            f"LS_KD: {self.ls_kd_meter.avg:.3f}, "
            f"{self._format_scalar_ra_details()}"
            f"{self._format_ugd_details()}"
            f"Acc: {self.acc_meter.avg:.3f}, "
            f"Base LR: {self.scheduler.get_last_lr()[0]:.2e}, "
            f"Time/Batch: {time_per_batch:.3f}s"
        )
