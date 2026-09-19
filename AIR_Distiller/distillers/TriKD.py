"""Compatibility imports for the original TriKD release."""

from .TriAIR import (
    DownSampling_Pooling,
    TriAIR,
    d3_loss,
    degradation_direction_distance,
    teacher_classifier_lsd_loss,
    weights_init_kaiming,
)

TriKD = TriAIR
