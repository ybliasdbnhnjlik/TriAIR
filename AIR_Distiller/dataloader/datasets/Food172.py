import os
import os.path as osp
from .bases import BaseImageDataset

class Food172(BaseImageDataset):
    """
    Food172 Dataset Loader with SOP-style query/gallery split.
    Random seed is used to ensure reproducibility.
    """

    dataset_dir = 'vireoFood-172'

    def __init__(self, root='./data/food-172', verbose=True, seed=42, **kwargs):
        super(Food172, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.train_list = osp.join(self.dataset_dir, 'train_full.txt')
        self.test_list = osp.join(self.dataset_dir, 'test_full.txt')
        self.seed = seed

        self._check_before_run()

        train = self._process_list(self.train_list, relabel=True)

        query_list, gallery_list = self._split_test_set(seed=self.seed)
        query = self._process_list_from_list(query_list, relabel=False, camid=0)
        gallery = self._process_list_from_list(gallery_list, relabel=False, camid=1)

        if verbose:
            print("=> Food172 loaded (SOP-style split with seed {})".format(self.seed))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(gallery)

        query_pids = set([pid for _, pid, _ in query])
        gallery_pids = set([pid for _, pid, _ in gallery])
        missing = query_pids - gallery_pids
        if missing:
            print(f"Query identities missing from gallery: {missing}")

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir): raise RuntimeError(f"{self.dataset_dir} not found")
        if not osp.exists(self.image_dir): raise RuntimeError(f"{self.image_dir} not found")
        if not osp.exists(self.train_list): raise RuntimeError(f"{self.train_list} not found")
        if not osp.exists(self.test_list): raise RuntimeError(f"{self.test_list} not found")

    def _process_list(self, txt_path, relabel=False):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        pid_container = set()
        for line in lines:
            img_rel_path, label = line.strip().split()
            pid_container.add(label)
        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            img_rel_path, label = line.strip().split()
            full_path = osp.join(self.image_dir, img_rel_path)
            pid = pid2label[label] if relabel else int(label)
            camid = 0
            dataset.append((full_path, pid, camid))
        return dataset

    def _split_test_set(self, seed=42):
        """
        Dynamically split the test set into SOP-style query and gallery sets.
        Randomly select one image per class as query, rest go to gallery.
        A fixed random seed is used to ensure reproducibility.
        """
        import random

        random.seed(seed)

        with open(self.test_list, 'r') as f:
            lines = f.readlines()

        pid_to_imgs = dict()
        for line in lines:
            line = line.strip()
            pid = line.split()[1]
            pid_to_imgs.setdefault(pid, []).append(line)

        query_list = []
        gallery_list = []

        for pid, imgs in pid_to_imgs.items():
            selected = random.choice(imgs)
            query_list.append(selected)
            gallery_list.extend([img for img in imgs if img != selected])

        return query_list, gallery_list

    def _process_list_from_list(self, lines, relabel=False, camid=0):
        pid_container = set()
        for line in lines:
            img_rel_path, label = line.strip().split()
            pid_container.add(label)

        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            img_rel_path, label = line.strip().split()
            full_path = osp.join(self.image_dir, img_rel_path)
            pid = pid2label[label] if relabel else int(label)
            dataset.append((full_path, pid, camid))

        return dataset
