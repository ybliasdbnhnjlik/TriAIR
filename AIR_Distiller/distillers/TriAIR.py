import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class DownSampling_Pooling(nn.Module):
    def __init__(self, size):
        super(DownSampling_Pooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        x = self.pooling(x)
        x_flatten = torch.flatten(x, start_dim=2, end_dim=3)

        x_flatten = x_flatten.permute((0, 2, 1))
        m, b, c = x_flatten.shape[0], x_flatten.shape[1], x_flatten.shape[2]

        x_flatten = torch.reshape(x_flatten, (m * b, c))
        return x_flatten


# ===== D3still 的关系蒸馏损失 =====
def d3_loss(student_features, teacher_features, topk, alpha, beta, gamma):
    batch_size = student_features.shape[0]

    # Normalize student and teacher features
    student_features = F.normalize(student_features, p=2, dim=1)
    teacher_features = F.normalize(teacher_features, p=2, dim=1)

    # Compute similarity matrices
    teacher_similarity = teacher_features.double().mm(teacher_features.double().t())
    cross_similarity = student_features.double().mm(teacher_features.double().t())

    # Get sorted teacher similarity and corresponding cross similarity
    teacher_topk_values, sorted_indices = torch.sort(teacher_similarity, dim=1, descending=True)
    student_topk_values = torch.gather(cross_similarity, 1, sorted_indices)

    # Feature distillation loss (alpha term): top-1 相似度对齐
    fd_loss = alpha * torch.norm(
        teacher_topk_values[:, 0] - student_topk_values[:, 0],
        p=2
    ) / batch_size

    # Extract top-k similarities (excluding the highest)
    student_distances = student_topk_values[:, 1:topk]
    teacher_distances = teacher_topk_values[:, 1:topk]

    # Compute pairwise difference matrices
    student_diff_matrix = student_distances.unsqueeze(1) - student_distances.unsqueeze(2)
    teacher_diff_matrix = teacher_distances.unsqueeze(1) - teacher_distances.unsqueeze(2)

    # Flatten the difference matrices
    student_diff_flat = student_diff_matrix.view(batch_size, -1)
    teacher_diff_flat = teacher_diff_matrix.view(batch_size, -1)

    # Avoid division by zero
    teacher_diff_flat[teacher_diff_flat == 0] = 1

    # Compute hard and simple weights
    hard_weights = (student_diff_flat / teacher_diff_flat).detach()
    simple_weights = (student_diff_flat / teacher_diff_flat).detach()

    hard_weights[hard_weights >= 0] = 0
    hard_weights[hard_weights < 0] = 1

    simple_weights[simple_weights <= 0] = 0
    simple_weights[simple_weights > 0] = 1

    # Avoid division by zero in student differences
    student_diff_flat[student_diff_flat == 0] = 1

    # Compute weighted result matrices
    hard_loss_matrix = hard_weights * (
        (student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs())
    )
    simple_loss_matrix = simple_weights * (
        (student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs())
    )

    # Relation KD loss (beta and gamma terms)
    hard_rd_loss = beta * torch.mean(torch.norm(hard_loss_matrix, p=2, dim=1)) / (topk - 1)
    simple_rd_loss = gamma * torch.mean(torch.norm(simple_loss_matrix, p=2, dim=1)) / (topk - 1)

    return fd_loss + hard_rd_loss + simple_rd_loss


def teacher_classifier_lsd_loss(
    student_features,
    teacher_features,
    teacher_classifier,
    tau=2.0,
    eps=1e-6,
):
    """Standardized class KL in the frozen teacher's semantic coordinates.

    Both inputs are retrieval features, already projected by their own model.
    Reuse only the teacher's final linear classifier, not its full FC/BN block.
    Its detached weights still propagate gradients into the student features.
    This replaces candidate-list R-LSD; it adds no classifier or ranking loss.
    """
    if student_features.ndim != 2 or teacher_features.ndim != 2:
        raise ValueError("student_features and teacher_features must be 2-D tensors")
    if student_features.shape != teacher_features.shape:
        raise ValueError("student and teacher retrieval features must have the same shape")
    if tau <= 0:
        raise ValueError("LSD temperature must be positive")

    # Keep the linear projections, standardization and KL in FP32 under AMP.
    # No feature L2 normalization: standardize the resulting class logits.
    with torch.autocast(device_type=student_features.device.type, enabled=False):
        weight = teacher_classifier.weight.detach().float()
        bias = teacher_classifier.bias
        if bias is not None:
            bias = bias.detach().float()

        def standardized_logits(features):
            logits = F.linear(features.float(), weight, bias)
            centered = logits - logits.mean(dim=1, keepdim=True)
            variance = centered.square().mean(dim=1, keepdim=True)
            return centered / variance.add(eps).sqrt()

        with torch.no_grad():
            teacher_z = standardized_logits(teacher_features.detach())
            teacher_probs = F.softmax(teacher_z / tau, dim=1)
        student_z = standardized_logits(student_features)
        return tau ** 2 * F.kl_div(
            F.log_softmax(student_z / tau, dim=1),
            teacher_probs,
            reduction="batchmean",
        )



def degradation_direction_distance(student, teacher_high, teacher_low, rho=0.25, eps=1e-6):
    """Per-region distance on L2-normalized features in one retrieval space.

    With e = student - teacher_high and delta = teacher_high - teacher_low,
    distance**2 = ||e||**2 - rho * <e, delta>**2 / (||delta||**2 + eps).
    Teacher targets/directions are detached. Only the local metric changes;
    the high-resolution teacher remains the target, including along delta.
    """
    if student.ndim != 2 or student.shape != teacher_high.shape or student.shape != teacher_low.shape:
        raise ValueError("Directional RA requires matching [regions, features] tensors")
    if not 0.0 <= rho < 1.0:
        raise ValueError("Directional RA rho must satisfy 0 <= rho < 1")
    if not eps > 0:
        raise ValueError("Directional RA eps must be positive")

    with torch.autocast(device_type=student.device.type, enabled=False):
        high = teacher_high.detach().float()
        delta = high - teacher_low.detach().float()
        error = student.float() - high
        u = delta / (delta.square().sum(dim=1, keepdim=True) + eps).sqrt()
        q = u.square().sum(dim=1, keepdim=True).clamp(max=1.0)
        # Square root of I - rho*u*u^T, applied without a dense matrix.
        # A vector norm gives a finite zero gradient at error=0, unlike a
        # direct sqrt(sum(error**2) - ...) implementation at that boundary.
        contraction = rho / (1.0 + (1.0 - rho * q).sqrt())
        adjusted_error = error - contraction * (error * u).sum(dim=1, keepdim=True) * u
        return torch.linalg.vector_norm(adjusted_error, dim=1)


class TriAIR(Distiller):
    r"""
    TriAIR for asymmetric retrieval.

    RSD transfers batch-level rankings through D3's similarity differences.
    DMGD uses global alignment and directional, aligned local supervision.
    LSD aligns standardized class distributions through the frozen teacher head.
    """

    def __init__(self, student, teacher, cfg):
        super(TriAIR, self).__init__(student, teacher, cfg)

        # ---------- UGD 部分超参 ----------
        self.distillation_layer = cfg.UGD.DISTILLATION_LAYER
        self.alpha_ugd = cfg.UGD.ALPHA
        self.beta_ugd = cfg.UGD.BETA
        self.kd_ugd_weight = cfg.UGD.KD_WEIGHT
        self.ra_enabled = getattr(cfg.UGD, "RA_ENABLED", False)
        # Saved configs without RA_MODE retain their scalar RA behavior.
        self.ra_mode = getattr(cfg.UGD, "RA_MODE", "scalar")
        self.ra_direction_rho = float(getattr(cfg.UGD, "RA_DIRECTION_RHO", 0.25))
        if self.ra_mode not in ("scalar", "directional"):
            raise ValueError("UGD.RA_MODE must be scalar or directional")
        if not 0.0 <= self.ra_direction_rho < 1.0:
            raise ValueError("UGD.RA_DIRECTION_RHO must satisfy 0 <= rho < 1")
        legacy_ra_tau = float(getattr(cfg.UGD, "RA_TAU", 0.2))
        fine_tau = getattr(cfg.UGD, "RA_FINE_TAU", None)
        coarse_tau = getattr(cfg.UGD, "RA_COARSE_TAU", None)
        self.ra_fine_tau = legacy_ra_tau if fine_tau is None else float(fine_tau)
        self.ra_coarse_tau = (
            legacy_ra_tau if coarse_tau is None else float(coarse_tau)
        )
        self.ra_fine_floor = float(getattr(cfg.UGD, "RA_FINE_FLOOR", 0.75))
        self.ra_coarse_floor = float(
            getattr(cfg.UGD, "RA_COARSE_FLOOR", 0.50)
        )
        self.ra_min_weight = float(getattr(cfg.UGD, "RA_MIN_WEIGHT", 0.0))
        self.ra_max_weight = float(getattr(cfg.UGD, "RA_MAX_WEIGHT", 1.0))
        self.ra_use_class_mask = getattr(cfg.UGD, "RA_USE_CLASS_MASK", True)
        self.ra_blend = float(getattr(cfg.UGD, "RA_BLEND", 0.25))
        self.ra_budget_normalize = bool(
            getattr(cfg.UGD, "RA_BUDGET_NORMALIZE", True)
        )
        self.ra_antialias = bool(getattr(cfg.UGD, "RA_ANTIALIAS", True))
        self.ra_accessibility_space = getattr(
            cfg.UGD, "RA_ACCESSIBILITY_SPACE", "backbone"
        )
        if self.ra_accessibility_space not in ("backbone", "retrieval"):
            raise ValueError("UGD.RA_ACCESSIBILITY_SPACE must be backbone or retrieval")

        if self.ra_fine_tau <= 0 or self.ra_coarse_tau <= 0:
            raise ValueError(
                "UGD.RA_FINE_TAU and UGD.RA_COARSE_TAU must be greater "
                "than zero"
            )
        if not 0.0 <= self.ra_fine_floor <= 1.0:
            raise ValueError("UGD.RA_FINE_FLOOR must be in [0, 1]")
        if not 0.0 <= self.ra_coarse_floor <= 1.0:
            raise ValueError("UGD.RA_COARSE_FLOOR must be in [0, 1]")
        if not 0.0 <= self.ra_min_weight <= self.ra_max_weight <= 1.0:
            raise ValueError(
                "UGD RA weights must satisfy 0 <= RA_MIN_WEIGHT <= "
                "RA_MAX_WEIGHT <= 1"
            )
        if not 0.0 <= self.ra_blend <= 1.0:
            raise ValueError("UGD.RA_BLEND must be in [0, 1]")

        pooling_size = cfg.INPUT.STUDENT_SIZE_TRAIN[0] // 16

        self.downsampling_pooling = DownSampling_Pooling((pooling_size, pooling_size))
        self.downsampling_pooling_half = DownSampling_Pooling((pooling_size // 2, pooling_size // 2))
        self.num_bottleneck = 512

        model_channel_num = {
            "ResNet18": [64, 128, 256, 512],
            "MobileNetV3_Small": [16, 24, 48, 576],
        }

        channel_num = model_channel_num[cfg.DISTILLER.STUDENT_NAME][self.distillation_layer - 1]

        self.projector = nn.Sequential(
            nn.BatchNorm1d(channel_num),
            nn.Linear(channel_num, self.num_bottleneck, bias=False),
            nn.BatchNorm1d(self.num_bottleneck)
        )
        self.projector.apply(weights_init_kaiming)

        self.projector_half = nn.Sequential(
            nn.BatchNorm1d(channel_num),
            nn.Linear(channel_num, self.num_bottleneck, bias=False),
            nn.BatchNorm1d(self.num_bottleneck)
        )
        self.projector_half.apply(weights_init_kaiming)

        # ---------- D3 部分超参 ----------
        self.topk = cfg.D3.TOPK
        self.alpha_d3 = cfg.D3.ALPHA
        self.beta_d3 = cfg.D3.BETA
        self.gamma_d3 = cfg.D3.GAMMA
        self.kd_d3_weight = cfg.D3.KD_WEIGHT

        # ---------- Frozen-teacher-classifier LSD 超参 ----------
        # Preserve the weight/temperature of saved R-LSD YAML files.
        lsd_weight = getattr(cfg.UGD, "LSD_WEIGHT", None)
        lsd_tau = getattr(cfg.UGD, "LSD_TAU", None)
        self.lsd_weight = float(
            getattr(cfg.UGD, "R_LSD_WEIGHT", 0.1) if lsd_weight is None else lsd_weight
        )
        self.lsd_tau = float(
            getattr(cfg.UGD, "R_LSD_TAU", 2.0) if lsd_tau is None else lsd_tau
        )
        if self.lsd_tau <= 0:
            raise ValueError("UGD.LSD_TAU must be positive")

    def get_learnable_parameters(self):
        return (super().get_learnable_parameters()
                + list(self.projector.named_parameters())
                + list(self.projector_half.named_parameters()))

    # ===== LSD：冻结教师分类器下的标准化类别蒸馏 =====
    def semantic_lsd_loss(self, teacher_features, student_features):
        return teacher_classifier_lsd_loss(
            student_features=student_features,
            teacher_features=teacher_features,
            teacher_classifier=self.teacher.fc.classifier,
            tau=self.lsd_tau,
        )

    def _resolution_accessibility(
        self,
        teacher_high_local,
        teacher_low_local,
        tau,
    ):
        """Estimate which teacher regions survive the student's resolution."""
        if teacher_high_local.shape != teacher_low_local.shape:
            raise ValueError(
                "High/low teacher local features must have identical shapes, got "
                f"{teacher_high_local.shape} and {teacher_low_local.shape}"
            )

        # Float32 keeps cosine/exp stable under CUDA autocast.  The result is
        # detached by construction: accessibility is a training-only teacher
        # estimate, not an optimization target.
        with torch.no_grad():
            high = F.normalize(teacher_high_local.detach().float(), p=2, dim=1)
            low = F.normalize(teacher_low_local.detach().float(), p=2, dim=1)
            cosine = (high * low).sum(dim=1).clamp(min=-1.0, max=1.0)
            accessibility = torch.exp(-(1.0 - cosine) / tau)
            accessibility = accessibility.clamp(
                min=self.ra_min_weight,
                max=self.ra_max_weight,
            )
        return accessibility, cosine

    @staticmethod
    def _residual_accessibility(accessibility, floor):
        """Keep a semantic supervision floor and use accessibility as a bonus.

        UGD's class mask has already established that a region is semantically
        useful.  RA-UGD v3 therefore modulates, rather than vetoes, that
        supervision: w = floor + (1 - floor) * accessibility.
        """
        return floor + (1.0 - floor) * accessibility

    @staticmethod
    def _masked_mean(value, valid_mask):
        valid_weight = valid_mask.to(dtype=value.dtype)
        return (value * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    @classmethod
    def _budget_normalized_weight(cls, weight, valid_mask, eps=1e-6):
        """Keep mean coefficients at one, not the loss or gradient magnitude."""
        if not valid_mask.any():
            return torch.ones_like(weight)
        mean_weight = cls._masked_mean(weight, valid_mask)
        # A zero floor and zero accessibility are legal configuration values.
        if mean_weight <= eps:
            return torch.ones_like(weight)
        return weight / mean_weight

    def _aligned_student_view(self, teacher_image, student_size):
        return F.interpolate(
            teacher_image,
            size=student_size,
            mode="bilinear",
            align_corners=False,
            antialias=self.ra_antialias,
        )

    @staticmethod
    def _forward_with_batch_stats_no_update(module, image):
        """Use batch BN statistics without modifying inference-time buffers.

        Disabling ``track_running_stats`` temporarily keeps training-mode batch
        normalization while leaving running_mean/running_var untouched.  This
        avoids using ordinary-view running statistics for the auxiliary view.
        """
        batch_norm_states = []
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                batch_norm_states.append(
                    (child, child.training, child.track_running_stats)
                )
                child.train(True)
                child.track_running_stats = False
        try:
            return module(image)
        finally:
            for child, was_training, was_tracking in batch_norm_states:
                child.track_running_stats = was_tracking
                child.train(was_training)

    @staticmethod
    def _fixed_denominator_local_loss(distance, valid_mask, accessibility):
        """Average a positive region-weighted local distance."""
        valid_weight = valid_mask.to(dtype=distance.dtype)
        accessibility = accessibility.to(dtype=distance.dtype)
        denominator = valid_weight.sum().clamp_min(1.0)
        return (distance * valid_weight * accessibility).sum() / denominator

    # ===== 独立全局视图 + 对齐局部 RA（标量/退化方向距离） =====
    def ugd_loss(
        self,
        kd_feature_teacher,
        kd_feature_student,
        target,
        kd_feature_teacher_low=None,
        kd_feature_student_ra=None,
    ):
        directional = self.ra_enabled and self.ra_mode == "directional"
        # teacher 局部特征（不反传）
        teacher_backbone = kd_feature_teacher["feats"][-1].detach()
        t_local = self.downsampling_pooling(teacher_backbone)
        t_local_score, t_local_feat = self.teacher.fc(t_local)

        t_local_half = self.downsampling_pooling_half(teacher_backbone)
        t_local_half_score, t_local_half_feat = self.teacher.fc(t_local_half)

        if self.ra_enabled:
            if kd_feature_teacher_low is None:
                raise ValueError("RA-UGD requires the low-resolution teacher features")
            teacher_backbone_low = kd_feature_teacher_low["feats"][-1].detach()
            t_local_low = self.downsampling_pooling(teacher_backbone_low)
            t_local_half_low = self.downsampling_pooling_half(teacher_backbone_low)
            high_fine, low_fine = t_local, t_local_low
            high_coarse, low_coarse = t_local_half, t_local_half_low
            if directional or self.ra_accessibility_space == "retrieval":
                # Directional distances always use the same 512-D retrieval
                # coordinates as the student projector and high-res target.
                with torch.no_grad():
                    _, low_fine = self.teacher.fc(t_local_low)
                    _, low_coarse = self.teacher.fc(t_local_half_low)
                high_fine, high_coarse = t_local_feat, t_local_half_feat
            if not directional:
                local_accessibility, local_cosine = self._resolution_accessibility(
                    high_fine,
                    low_fine,
                    self.ra_fine_tau,
                )
                half_accessibility, half_cosine = self._resolution_accessibility(
                    high_coarse,
                    low_coarse,
                    self.ra_coarse_tau,
                )
        else:
            local_accessibility = torch.ones(
                t_local.size(0), device=t_local.device, dtype=torch.float32
            )
            half_accessibility = torch.ones(
                t_local_half.size(0), device=t_local_half.device, dtype=torch.float32
            )
            local_cosine = local_accessibility
            half_cosine = half_accessibility

        # Local UGD must compare corresponding cells.  When RA is enabled, both
        # the unweighted base term and accessibility modulation therefore use
        # the aligned student branch.  The independent student branch remains
        # responsible for global UGD, D3 and semantic LSD only.
        if self.ra_enabled:
            if kd_feature_student_ra is None:
                raise ValueError("RA-UGD requires aligned student RA features")
            s_local = self.downsampling_pooling(
                kd_feature_student_ra["feats"][self.distillation_layer - 1]
            )
            s_local_half = self.downsampling_pooling_half(
                kd_feature_student_ra["feats"][self.distillation_layer - 1]
            )
        else:
            s_local = self.downsampling_pooling(
                kd_feature_student["feats"][self.distillation_layer - 1]
            )
            s_local_half = self.downsampling_pooling_half(
                kd_feature_student["feats"][self.distillation_layer - 1]
            )

        # Projectors are training-only modules.  They see exactly one local
        # branch per iteration and can safely use/update their own BN stats.
        s_local_feat = self.projector(s_local)
        s_local_half_feat = self.projector_half(s_local_half)

        # 全局检索特征对齐
        s_feat = F.normalize(kd_feature_student["retrieval_feat"], p=2, dim=1)
        t_feat = F.normalize(kd_feature_teacher["retrieval_feat"].detach(), p=2, dim=1)
        f_loss = torch.norm(t_feat - s_feat, p=2, dim=1).mean()

        # Direction estimation and local normalization stay in FP32 under AMP.
        # Leave the scalar path unchanged for existing experiment comparisons.
        if directional:
            with torch.autocast(device_type=s_local_feat.device.type, enabled=False):
                t_local_feat = F.normalize(t_local_feat.detach().float(), p=2, dim=1)
                t_local_half_feat = F.normalize(t_local_half_feat.detach().float(), p=2, dim=1)
                low_fine = F.normalize(low_fine.detach().float(), p=2, dim=1)
                low_coarse = F.normalize(low_coarse.detach().float(), p=2, dim=1)
                s_local_feat = F.normalize(s_local_feat.float(), p=2, dim=1)
                s_local_half_feat = F.normalize(s_local_half_feat.float(), p=2, dim=1)
        else:
            t_local_feat = F.normalize(t_local_feat, p=2, dim=1)
            t_local_half_feat = F.normalize(t_local_half_feat, p=2, dim=1)
            s_local_feat = F.normalize(s_local_feat, p=2, dim=1)
            s_local_half_feat = F.normalize(s_local_half_feat, p=2, dim=1)

        # 展开标签
        local_target = target.repeat_interleave(int(s_local_feat.size(0) / len(target)))
        local_half_target = target.repeat_interleave(int(s_local_half_feat.size(0) / len(target)))

        # teacher 局部分类预测，用于筛选“语义明确”的区域
        t_local_predict = torch.argmax(t_local_score, dim=1)
        t_local_half_predict = torch.argmax(t_local_half_score, dim=1)

        if self.ra_use_class_mask:
            local_true_mask = t_local_predict == local_target
            local_true_half_mask = t_local_half_predict == local_half_target
        else:
            # Label-free ablation: keep every local region.
            local_true_mask = torch.ones_like(t_local_predict, dtype=torch.bool)
            local_true_half_mask = torch.ones_like(
                t_local_half_predict, dtype=torch.bool
            )

        local_distance = torch.norm(t_local_feat - s_local_feat, p=2, dim=1)
        local_half_distance = torch.norm(
            t_local_half_feat - s_local_half_feat, p=2, dim=1
        )

        if directional:
            # Replace scalar region reweighting, rather than adding a loss or
            # multiplying by the old tau/floor/blend-based weights.
            local_direction_distance = degradation_direction_distance(
                s_local_feat, t_local_feat, low_fine, self.ra_direction_rho,
            )
            half_direction_distance = degradation_direction_distance(
                s_local_half_feat, t_local_half_feat, low_coarse, self.ra_direction_rho,
            )
            local_true_loss = self._masked_mean(local_direction_distance, local_true_mask)
            local_true_half_loss = self._masked_mean(half_direction_distance, local_true_half_mask)
        elif self.ra_enabled:
            local_ra_weight = self._residual_accessibility(
                local_accessibility, self.ra_fine_floor,
            )
            half_ra_weight = self._residual_accessibility(
                half_accessibility, self.ra_coarse_floor,
            )
            if self.ra_budget_normalize:
                local_ra_weight = self._budget_normalized_weight(
                    local_ra_weight,
                    local_true_mask,
                )
                half_ra_weight = self._budget_normalized_weight(
                    half_ra_weight,
                    local_true_half_mask,
                )
            # Convex residual modulation on the same aligned distance avoids
            # the conflicting independent-vs-aligned objectives in v3.1.
            local_effective_weight = (
                (1.0 - self.ra_blend) + self.ra_blend * local_ra_weight
            )
            half_effective_weight = (
                (1.0 - self.ra_blend) + self.ra_blend * half_ra_weight
            )
        else:
            local_effective_weight = torch.ones_like(local_distance)
            half_effective_weight = torch.ones_like(local_half_distance)

        if not directional:
            local_true_loss = self._fixed_denominator_local_loss(
                local_distance,
                local_true_mask,
                local_effective_weight,
            )
            local_true_half_loss = self._fixed_denominator_local_loss(
                local_half_distance,
                local_true_half_mask,
                half_effective_weight,
            )

        loss = self.alpha_ugd * f_loss + self.beta_ugd * (local_true_loss + local_true_half_loss)
        diagnostics = {
            "ugd_global": f_loss.detach(),
            "ugd_fine": local_true_loss.detach(),
            "ugd_coarse": local_true_half_loss.detach(),
            "ra_fine_loss_delta": (
                local_true_loss - self._masked_mean(local_distance, local_true_mask)
            ).detach(),
            "ra_coarse_loss_delta": (
                local_true_half_loss
                - self._masked_mean(local_half_distance, local_true_half_mask)
            ).detach(),
        }
        if directional:
            with torch.no_grad():
                def distance_ratio(adjusted, original):
                    # Perfect matches have ratio 1 by convention, not 0/0.
                    return torch.where(
                        original > 0, adjusted / original.clamp_min(1e-12),
                        torch.ones_like(original),
                    )

                diagnostics.update({
                    # Counts let the trainer aggregate conditional diagnostics
                    # over valid regions, excluding empty batches/replicas.
                    "ra_direction_fine_valid_count": local_true_mask.sum(),
                    "ra_direction_coarse_valid_count": local_true_half_mask.sum(),
                    "ra_direction_fine_region_count": local_true_mask.new_tensor(
                        local_true_mask.numel(), dtype=torch.long,
                    ),
                    "ra_direction_coarse_region_count": local_true_half_mask.new_tensor(
                        local_true_half_mask.numel(), dtype=torch.long,
                    ),
                    "ra_direction_valid_ratio": torch.cat(
                        (local_true_mask, local_true_half_mask)
                    ).float().mean(),
                    "ra_direction_hr_lr_similarity": torch.cat((
                        (t_local_feat * low_fine).sum(1),
                        (t_local_half_feat * low_coarse).sum(1),
                    )).mean(),
                    "ra_direction_fine_norm": self._masked_mean(
                        torch.linalg.vector_norm(t_local_feat - low_fine, dim=1), local_true_mask,
                    ),
                    "ra_direction_coarse_norm": self._masked_mean(
                        torch.linalg.vector_norm(t_local_half_feat - low_coarse, dim=1), local_true_half_mask,
                    ),
                    "ra_direction_fine_ratio": self._masked_mean(
                        distance_ratio(local_direction_distance, local_distance), local_true_mask,
                    ),
                    "ra_direction_coarse_ratio": self._masked_mean(
                        distance_ratio(half_direction_distance, local_half_distance), local_true_half_mask,
                    ),
                })
            return loss, diagnostics

        diagnostics.update({
            "ra_accessibility": torch.cat(
                (local_accessibility, half_accessibility)
            ).mean().detach(),
            "ra_valid_ratio": torch.cat(
                (local_true_mask, local_true_half_mask)
            ).float().mean().detach(),
            "ra_hr_lr_similarity": torch.cat(
                (local_cosine, half_cosine)
            ).mean().detach(),
            "ra_fine_effective_weight": self._masked_mean(
                local_effective_weight,
                local_true_mask,
            ).detach(),
            "ra_coarse_effective_weight": self._masked_mean(
                half_effective_weight,
                local_true_half_mask,
            ).detach(),
            "ra_fine_weight_std": local_effective_weight[
                local_true_mask
            ].float().std(unbiased=False).detach()
            if local_true_mask.any()
            else local_effective_weight.new_zeros(()),
            "ra_coarse_weight_std": half_effective_weight[
                local_true_half_mask
            ].float().std(unbiased=False).detach()
            if local_true_half_mask.any()
            else half_effective_weight.new_zeros(()),
        })
        return loss, diagnostics

    # ===== 训练前向：CE + Triplet +（UGD KD + D3 KD + semantic LSD） =====
    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        # 主分支：学生正常训练
        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        # Main KD branch keeps the original independent augmentations.  This is
        # important for D3/LSD cross-view invariance.
        _, kd_feature_student = self.student(kd_student_image)
        kd_feature_student_ra = None
        ra_student_image = None
        if self.ra_enabled:
            # Downsample the teacher view itself so the auxiliary student grid
            # is exactly aligned with the high-resolution teacher grid.
            ra_student_image = self._aligned_student_view(
                kd_teacher_image,
                kd_student_image.shape[-2:],
            )
            _, kd_feature_student_ra = self._forward_with_batch_stats_no_update(
                self.student,
                ra_student_image,
            )

        with torch.no_grad():
            _, kd_feature_teacher = self.teacher(kd_teacher_image)
            kd_feature_teacher_low = None
            if self.ra_enabled:
                # Re-upsample the aligned low-resolution proxy for teachers
                # with a fixed input size (including Swin).
                teacher_low_image = F.interpolate(
                    ra_student_image,
                    size=kd_teacher_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                _, kd_feature_teacher_low = self.teacher(teacher_low_image)

        # RA-UGD 特征蒸馏
        ugd_loss, ra_diagnostics = self.ugd_loss(
            kd_feature_teacher,
            kd_feature_student,
            kd_target,
            kd_feature_teacher_low,
            kd_feature_student_ra,
        )
        ugd_kd = self.kd_ugd_weight * ugd_loss

        # D3 差分关系蒸馏（基于 retrieval_feat）
        d3_kd = self.kd_d3_weight * d3_loss(
            kd_feature_student["retrieval_feat"],
            kd_feature_teacher["retrieval_feat"],
            self.topk,
            self.alpha_d3,
            self.beta_d3,
            self.gamma_d3,
        )

        # Replace R-LSD with semantic KL in the frozen teacher's coordinates.
        # The student's own classifier remains trained by the original CE.
        ls_kd = self.lsd_weight * self.semantic_lsd_loss(
            kd_feature_teacher["retrieval_feat"].detach(),
            kd_feature_student["retrieval_feat"],
        )

        # loss_kd is the optimization term.  The two component entries below
        # are diagnostics only; the trainer explicitly excludes them when it
        # constructs the scalar objective.
        kd_loss = ugd_kd + d3_kd

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,
            "loss_kd_ugd": ugd_kd,
            "loss_kd_d3": d3_kd,
            "loss_LS_KD": ls_kd,
            **ra_diagnostics,
        }
        return logits_student, losses_dict
