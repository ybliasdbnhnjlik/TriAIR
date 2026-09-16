import torch
from torch.utils.data import DataLoader
from .datasets import CUB200, InShop, MSMT17, SOP,Food101,Food172
from .datasets.bases import ImageDataset, Distillation_ImageDataset
from .sampler import RandomIdentitySampler
import numpy as np
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class SynchronizedDistillationTransform:
    """Build spatially aligned student/teacher KD views.

    Flip, padded-crop location, and random-erasing rectangle are sampled once
    in normalized image coordinates and then mapped to the two resolutions.
    This makes grid cell ``i`` describe the same image area in both views.
    """

    def __init__(
        self,
        student_size,
        teacher_size,
        student_padding,
        teacher_padding,
        flip_probability,
        erasing_probability,
        pixel_mean,
        pixel_std,
    ):
        self.student_size = tuple(student_size)
        self.teacher_size = tuple(teacher_size)
        self.student_padding = student_padding
        self.teacher_padding = teacher_padding
        self.flip_probability = float(flip_probability)
        self.erasing_probability = float(erasing_probability)
        self.pixel_mean = list(pixel_mean)
        self.pixel_std = list(pixel_std)

    @staticmethod
    def _sample_aligned_offsets(
        student_source_size,
        student_crop_size,
        teacher_source_size,
        teacher_crop_size,
        position,
    ):
        student_max_offset = max(
            int(student_source_size) - int(student_crop_size), 0
        )
        teacher_max_offset = max(
            int(teacher_source_size) - int(teacher_crop_size), 0
        )
        student_offset = int(round(position * student_max_offset))

        # Quantize the crop at the deployed student resolution first, then
        # map that exact location to the teacher view.  With proportional
        # padding (2 vs. 8), this maps offsets exactly (e.g. 1 -> 4).
        if student_max_offset > 0:
            position = student_offset / student_max_offset
        teacher_offset = int(round(position * teacher_max_offset))
        return student_offset, teacher_offset

    @staticmethod
    def _scale_erasing_box(box, source_shape, target_shape):
        top, left, height, width = box
        source_height, source_width = source_shape
        target_height, target_width = target_shape

        scale_height = target_height / source_height
        scale_width = target_width / source_width
        target_top = int(round(top * scale_height))
        target_left = int(round(left * scale_width))
        target_box_height = max(1, int(round(height * scale_height)))
        target_box_width = max(1, int(round(width * scale_width)))

        target_top = min(target_top, target_height - 1)
        target_left = min(target_left, target_width - 1)
        target_box_height = min(target_box_height, target_height - target_top)
        target_box_width = min(target_box_width, target_width - target_left)
        return target_top, target_left, target_box_height, target_box_width

    def __call__(self, image):
        student_image = TF.resize(
            image,
            self.student_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        teacher_image = TF.resize(
            image,
            self.teacher_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        if random.random() < self.flip_probability:
            student_image = TF.hflip(student_image)
            teacher_image = TF.hflip(teacher_image)

        student_image = TF.pad(student_image, self.student_padding)
        teacher_image = TF.pad(teacher_image, self.teacher_padding)

        vertical_position = random.random()
        horizontal_position = random.random()
        student_top, teacher_top = self._sample_aligned_offsets(
            student_image.height,
            self.student_size[0],
            teacher_image.height,
            self.teacher_size[0],
            vertical_position,
        )
        student_left, teacher_left = self._sample_aligned_offsets(
            student_image.width,
            self.student_size[1],
            teacher_image.width,
            self.teacher_size[1],
            horizontal_position,
        )

        student_image = TF.crop(
            student_image,
            student_top,
            student_left,
            self.student_size[0],
            self.student_size[1],
        )
        teacher_image = TF.crop(
            teacher_image,
            teacher_top,
            teacher_left,
            self.teacher_size[0],
            self.teacher_size[1],
        )

        student_image = TF.normalize(TF.to_tensor(student_image), self.pixel_mean, self.pixel_std)
        teacher_image = TF.normalize(TF.to_tensor(teacher_image), self.pixel_mean, self.pixel_std)

        if self.erasing_probability > 0 and random.random() < self.erasing_probability:
            top, left, height, width, value = T.RandomErasing.get_params(
                student_image,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=self.pixel_mean,
            )
            # torchvision falls back to returning the whole input tensor as
            # ``value`` if no valid rectangle is found.  Such a tensor cannot
            # be reused at the teacher resolution, so preserve the configured
            # constant fill value in that rare case.
            if value.shape[-2:] != (1, 1):
                value = torch.as_tensor(
                    self.pixel_mean,
                    dtype=student_image.dtype,
                    device=student_image.device,
                ).view(-1, 1, 1)
            student_image = TF.erase(
                student_image, top, left, height, width, value, inplace=False
            )
            teacher_box = self._scale_erasing_box(
                (top, left, height, width),
                student_image.shape[-2:],
                teacher_image.shape[-2:],
            )
            teacher_image = TF.erase(
                teacher_image, *teacher_box, value, inplace=False
            )

        return student_image, teacher_image




class DataLoaderFactory:
    def __init__(self, cfg):

        self.cfg = cfg
        self.factory = {'CUB200': CUB200, 'InShop': InShop, 'SOP': SOP, 'MSMT17': MSMT17, 'Food101': Food101, 'Food172': Food172}

        # Define transforms
        self.s_train_transforms = self._build_train_transforms(self.cfg.INPUT.STUDENT_SIZE_TRAIN, self.cfg.INPUT.STUDENT_PADDING)
        self.t_train_transforms = self._build_train_transforms(self.cfg.INPUT.TEACHER_SIZE_TRAIN, self.cfg.INPUT.TEACHER_PADDING)
        self.kd_paired_transform = None
        if self.cfg.INPUT.KD_SYNC_AUGMENTATION:
            self.kd_paired_transform = SynchronizedDistillationTransform(
                student_size=self.cfg.INPUT.STUDENT_SIZE_TRAIN,
                teacher_size=self.cfg.INPUT.TEACHER_SIZE_TRAIN,
                student_padding=self.cfg.INPUT.STUDENT_PADDING,
                teacher_padding=self.cfg.INPUT.TEACHER_PADDING,
                flip_probability=self.cfg.INPUT.PROB,
                erasing_probability=self.cfg.INPUT.RE_PROB,
                pixel_mean=self.cfg.INPUT.PIXEL_MEAN,
                pixel_std=self.cfg.INPUT.PIXEL_STD,
            )
        self.query_transforms = self._build_test_transforms(self.cfg.INPUT.STUDENT_SIZE_TEST)
        self.gallery_transforms = self._build_test_transforms(self.cfg.INPUT.TEACHER_SIZE_TEST)

    def _build_train_transforms(self, size, padding):
        transforms = [
            T.Resize(size),
            T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
            T.Pad(padding),
            T.RandomCrop(size),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ]
        if self.cfg.INPUT.RE_PROB > 0:
            transforms.append(T.RandomErasing(p=self.cfg.INPUT.RE_PROB, value=self.cfg.INPUT.PIXEL_MEAN))
        return T.Compose(transforms)

    def _build_test_transforms(self, size):
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ])


    def _worker_init_fn(self, worker_id):
        """Worker init function to set random seeds."""
        np.random.seed(self.cfg.SOLVER.SEED + worker_id)
        random.seed(self.cfg.SOLVER.SEED + worker_id)

    @staticmethod
    def train_collate_fn(batch):
        img, pids, _, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        return torch.stack(img, dim=0), pids

    @staticmethod
    def distillation_train_collate_fn(batch):
        s_img, t_img, pids, _, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        return torch.stack(s_img, dim=0), torch.stack(t_img, dim=0), pids

    @staticmethod
    def val_collate_fn(batch):
        imgs, pids, camids, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids

    def create_dataloaders(self):

        if self.cfg.DATASETS.NAMES not in self.factory:
            raise ValueError(
                f"Dataset {self.cfg.DATASETS.NAMES} is not supported. "
                f"Available datasets: {list(self.factory.keys())}"
            )

        self.dataset = self.factory[self.cfg.DATASETS.NAMES](root=self.cfg.DATASETS.ROOT_DIR)
        self.num_classes = self.dataset.num_train_pids

        # Student train loader
        student_train_set = ImageDataset(self.dataset.train, self.s_train_transforms)
        if 'triplet' in self.cfg.DATALOADER.SAMPLER:
            print('Using triplet sampler')
            # Ensure IMS_PER_BATCH is divisible by NUM_INSTANCE
            if self.cfg.SOLVER.IMS_PER_BATCH % self.cfg.DATALOADER.NUM_INSTANCE != 0:
                raise ValueError(
                    f"cfg.SOLVER.IMS_PER_BATCH ({self.cfg.SOLVER.IMS_PER_BATCH}) must be divisible by "
                    f"cfg.DATALOADER.NUM_INSTANCE ({self.cfg.DATALOADER.NUM_INSTANCE}). Please adjust your configuration."
                )
            student_train_loader = DataLoader(
                student_train_set,
                batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    self.dataset.train,
                    self.cfg.SOLVER.IMS_PER_BATCH,
                    self.cfg.DATALOADER.NUM_INSTANCE
                ),
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                collate_fn=self.train_collate_fn,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn
            )

        elif self.cfg.DATALOADER.SAMPLER == 'random':
            print('Using random sampler')
            student_train_loader = DataLoader(
                student_train_set,
                batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                collate_fn=self.train_collate_fn,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=self._worker_init_fn
            )
        else:
            raise ValueError(
                f"Unsupported sampler: expected 'random' or 'triplet' but got {self.cfg.DATALOADER.SAMPLER}"
            )

        if self.cfg.DISTILLER.TYPE == "NONE":
            distillation_loader = None

        else:
            # Distillation loader
            distillation_train_set = Distillation_ImageDataset(
                self.dataset.train,
                self.s_train_transforms,
                self.t_train_transforms,
                paired_transform=self.kd_paired_transform,
            )

            distillation_loader = DataLoader(
                distillation_train_set,
                batch_size=self.cfg.SOLVER.IMS_DISTILLATION_PER_BATCH,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                collate_fn=self.distillation_train_collate_fn,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=self._worker_init_fn
            )

        query_loader, gallery_loader = self._evaluation_loaders()
        return student_train_loader, distillation_loader, query_loader, gallery_loader, self.num_classes

    def create_evaluation_dataloaders(self):
        """Build evaluation loaders without constructing training samplers."""
        if self.cfg.DATASETS.NAMES not in self.factory:
            raise ValueError(f"Unsupported dataset: {self.cfg.DATASETS.NAMES}")
        self.dataset = self.factory[self.cfg.DATASETS.NAMES](root=self.cfg.DATASETS.ROOT_DIR)
        self.num_classes = self.dataset.num_train_pids
        query_loader, gallery_loader = self._evaluation_loaders()
        return query_loader, gallery_loader, self.num_classes

    def _evaluation_loaders(self):
        gallery_transforms = (self.query_transforms if self.cfg.DISTILLER.TYPE == "NONE"
                              else self.gallery_transforms)
        gallery_set = ImageDataset(self.dataset.gallery, gallery_transforms)
        query_set = ImageDataset(self.dataset.query, self.query_transforms)
        query_loader = DataLoader(
            query_set,
            batch_size=self.cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.val_collate_fn
        )

        # Gallery loader

        gallery_loader = DataLoader(
            gallery_set,
            batch_size=self.cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            collate_fn=self.val_collate_fn
        )

        return query_loader, gallery_loader
