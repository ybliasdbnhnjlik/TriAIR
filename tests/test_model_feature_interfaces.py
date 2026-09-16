from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AIR_Distiller"))

from models import model_dict


class TeacherFeatureInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(2024)

    def test_swin_normalizes_hr_and_lr_local_features_without_changing_global_outputs(self):
        # Real configured backbone, no checkpoint/download dependency.
        teacher = model_dict["Swin_Transformer_V2_Small"](
            pretrained=False, num_classes=4,
        ).eval()
        high = torch.randn(1, 3, 256, 256)
        low = F.interpolate(high, size=(64, 64), mode="bilinear", align_corners=False, antialias=True)
        low = F.interpolate(low, size=(256, 256), mode="bilinear", align_corners=False)
        images = torch.cat((high, low))

        with torch.inference_mode():
            logits, features = teacher(images)
            old_final, old_intermediates = teacher.model.forward_intermediates(images, norm=False)
            old_logits, old_retrieval = teacher.fc(old_final.mean((1, 2)))
            expected_local = teacher.model.norm(
                old_intermediates[-1].permute(0, 2, 3, 1),
            ).permute(0, 3, 1, 2).contiguous()
            torch.testing.assert_close(features["feats"][-1], expected_local)
            self.assertFalse(torch.allclose(old_intermediates[-1], expected_local))
            for current, old in zip(features["feats"][:-1], old_intermediates[:-1]):
                torch.testing.assert_close(current, old)

            # The final global embedding/classifier path must stay unchanged.
            torch.testing.assert_close(logits, old_logits)
            torch.testing.assert_close(features["retrieval_feat"], old_retrieval)
            torch.testing.assert_close(features["pooled_feat"], old_final.mean((1, 2)))

            # 1x1 local pooling must recover the global FC input and output.
            pooled = F.adaptive_avg_pool2d(features["feats"][-1], 1).flatten(1)
            local_logits, local_retrieval = teacher.fc(pooled)
            torch.testing.assert_close(pooled, features["pooled_feat"])
            torch.testing.assert_close(local_logits, logits, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(local_retrieval, features["retrieval_feat"], atol=1e-6, rtol=1e-5)

    def test_resnet101_local_global_feature_contract(self):
        teacher = model_dict["ResNet101"](
            pretrained=False, last_stride=1, num_classes=4,
        ).eval()
        with torch.inference_mode():
            logits, features = teacher(torch.randn(2, 3, 64, 64))
            pooled = F.adaptive_avg_pool2d(features["feats"][-1], 1).flatten(1)
            local_logits, local_retrieval = teacher.fc(pooled)
            torch.testing.assert_close(pooled, features["pooled_feat"])
            torch.testing.assert_close(local_logits, logits)
            torch.testing.assert_close(local_retrieval, features["retrieval_feat"])


if __name__ == "__main__":
    unittest.main()
