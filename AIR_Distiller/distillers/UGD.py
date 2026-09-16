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



class UGD(Distiller):
    """Unambiguous granularity distillation for asymmetric image retrieval. Neural Networks 2025"""

    def __init__(self, student, teacher, cfg):
        super(UGD, self).__init__(student, teacher, cfg)

        self.distillation_layer = cfg.UGD.DISTILLATION_LAYER
        self.alpha = cfg.UGD.ALPHA
        self.beta = cfg.UGD.BETA
        self.kd_loss_weight = cfg.UGD.KD_WEIGHT

        # ===== 新增：LS-KD 的超参数（可在 cfg.UGD 里配） =====
        # 没配的话就用默认：权重 1.0，温度 2.0
        self.ls_kd_weight = getattr(cfg.UGD, "LS_KD_WEIGHT",0.1)
        self.ls_tau = getattr(cfg.UGD, "LS_TAU", 2.0)

        pooling_size = cfg.INPUT.STUDENT_SIZE_TRAIN[0] // 16

        self.downsampling_pooling = DownSampling_Pooling((pooling_size, pooling_size))
        self.downsampling_pooling_half = DownSampling_Pooling((pooling_size // 2, pooling_size // 2))
        self.num_bottleneck = 512

        model_channel_num = {
            "ResNet18": [64, 128, 256, 512],
            "MobileNetV3_Small": [16, 24, 48, 576]
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

    def get_learnable_parameters(self):
        return (super().get_learnable_parameters()
                + list(self.projector.named_parameters())
                + list(self.projector_half.named_parameters()))

    # ===== 新增：LS-KD 用的 Z-score 标准化 =====
    @staticmethod
    def _z_score_logits(logits, tau, eps=1e-6):
        """
        对每个样本的 logits 做标准化：
        Z(x; tau) = (x - mean) / (std * tau)
        """
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True).clamp_min(eps)
        return (logits - mean) / (std * tau)

    # ===== 新增：LS-KD 损失 =====
    def ls_kd_loss(self, t_logits, s_logits):
        """
        Logit Standardization KD:
        1. 先对 teacher / student 的 logits 做 Z-score 标准化
        2. 再算 KL(student || teacher)，最后乘 tau^2
        """
        with torch.no_grad():
            t_z = self._z_score_logits(t_logits, self.ls_tau)
            p_t = F.softmax(t_z, dim=1)

        s_z = self._z_score_logits(s_logits, self.ls_tau)
        log_p_s = F.log_softmax(s_z, dim=1)

        loss = F.kl_div(log_p_s, p_t, reduction='batchmean')
        loss = (self.ls_tau ** 2) * loss
        return loss

    def forward_train(self, image, kd_student_image, kd_teacher_image, target, kd_target, **kwargs):

        logits_student, feature_student = self.student(image)
        ce_loss = self.ce_loss_weight * self.ce_loss(logits_student, target)
        triplet_loss = self.tri_loss_weight * self.triplet_loss(feature_student["pooled_feat"], target)

        # 原来这里丢掉了 logits，这里保留出来，给 LS-KD 用
        kd_logits_student, kd_feature_student = self.student(kd_student_image)
        kd_logits_teacher, kd_feature_teacher = self.teacher(kd_teacher_image)

        # 原始 UGD 的 KD（特征蒸馏），保持不变
        kd_loss = self.kd_loss_weight * self.ugd_loss(
            kd_feature_teacher, kd_feature_student, kd_target
        )

        # 新增：LS-KD 损失（logits 蒸馏）
        ls_kd_loss = self.ls_kd_weight * self.ls_kd_loss(
            kd_logits_teacher.detach(),  # 教师不回传梯度
            kd_logits_student
        )

        losses_dict = {
            "loss_ce": ce_loss,
            "loss_triplet": triplet_loss,
            "loss_kd": kd_loss,          # 原来的 KD（UGD）
            "loss_LS_KD": ls_kd_loss,    # 新增的 LS-KD
        }
        return logits_student, losses_dict


    def ugd_loss(self, kd_feature_teacher, kd_feature_student, target):

        t_local = self.downsampling_pooling(kd_feature_teacher["feats"][-1].detach())
        t_local_score, t_local_feat = self.teacher.fc(t_local)

        t_local_half = self.downsampling_pooling_half(kd_feature_teacher["feats"][-1].detach())
        t_local_half_score, t_local_half_feat = self.teacher.fc(t_local_half)

        s_local = self.downsampling_pooling(kd_feature_student["feats"][self.distillation_layer - 1])
        s_local_feat = self.projector(s_local)

        s_local_half = self.downsampling_pooling_half(kd_feature_student["feats"][self.distillation_layer - 1])
        s_local_half_feat = self.projector_half(s_local_half)

        s_feat = F.normalize(kd_feature_student["retrieval_feat"], p=2, dim=1)
        t_feat = F.normalize(kd_feature_teacher["retrieval_feat"].detach(), p=2, dim=1)

        f_loss = torch.norm(t_feat - s_feat, p=2, dim=1).mean()

        t_local_feat = F.normalize(t_local_feat, p=2, dim=1)
        t_local_half_feat = F.normalize(t_local_half_feat, p=2, dim=1)

        s_local_feat = F.normalize(s_local_feat, p=2, dim=1)
        s_local_half_feat = F.normalize(s_local_half_feat, p=2, dim=1)

        local_target = target.repeat_interleave(int(s_local_feat.size(0) / len(target)))
        local_half_target = target.repeat_interleave(int(s_local_half_feat.size(0) / len(target)))

        t_local_predict = torch.argmax(t_local_score, dim=1)
        t_local_half_predict = torch.argmax(t_local_half_score, dim=1)

        local_true_mask = t_local_predict == local_target
        local_true_half_mask = t_local_half_predict == local_half_target

        if local_true_mask.any():
            local_true_loss = torch.norm(
                t_local_feat - s_local_feat, p=2, dim=1
            )[local_true_mask].mean()
        else:
            local_true_loss = torch.tensor(0.0, device=t_local_feat.device)

        if local_true_half_mask.any():
            local_true_half_loss = torch.norm(
                t_local_half_feat - s_local_half_feat, p=2, dim=1
            )[local_true_half_mask].mean()
        else:
            local_true_half_loss = torch.tensor(0.0, device=t_local_half_feat.device)

        loss = self.alpha * f_loss + self.beta * (local_true_loss + local_true_half_loss)

        return loss
