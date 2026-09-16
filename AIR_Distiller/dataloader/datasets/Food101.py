import os
import os.path as osp
from .bases import BaseImageDataset

class Food101(BaseImageDataset):
    """
    Food101 Dataset Loader with SOP-style query/gallery split.
    - 101 categories
    - 75,750 training images (750 per class)
    - 25,250 test images (250 per class)
    """

    dataset_dir = 'food-101'

    def __init__(self, root='./data/food-101', verbose=True, **kwargs):
        super(Food101, self).__init__()
        self.dataset_dir = osp.join(root)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'meta', 'train.txt')
        self.test_list = osp.join(self.dataset_dir, 'meta', 'test.txt')

        self._check_before_run()

        train = self._process_list(self.train_list, relabel=True)

        query_list, gallery_list = self._split_test_set()
        query = self._process_list_from_list(query_list, relabel=False, camid=0)
        gallery = self._process_list_from_list(gallery_list, relabel=False, camid=1)

        if verbose:
            print("=> Food101 loaded (SOP-style split)")
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
            pid = line.strip().split('/')[0]
            pid_container.add(pid)
        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            img_rel_path = line.strip()
            full_path = osp.join(self.image_dir, img_rel_path + '.jpg')
            pid = img_rel_path.split('/')[0]
            pid = pid2label[pid] if relabel or pid in pid2label else pid
            camid = 0
            dataset.append((full_path, pid, camid))
        return dataset

    def _split_test_set(self):
        """
        Split the test set into SOP-style query and gallery sets.
        One image per class in query, rest in gallery.
        """
        with open(self.test_list, 'r') as f:
            lines = f.readlines()

        pid_to_imgs = dict()
        for line in lines:
            line = line.strip()
            pid = line.split('/')[0]
            pid_to_imgs.setdefault(pid, []).append(line)

        query_list = []
        gallery_list = []

        for pid, imgs in pid_to_imgs.items():
            imgs = sorted(imgs)
            query_list.append(imgs[0])
            gallery_list.extend(imgs[1:])

        return query_list, gallery_list
    def _process_list_from_list(self, lines, relabel=False, camid=0):
        pid_container = set()
        for line in lines:
            pid = line.strip().split('/')[0]
            pid_container.add(pid)

        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            img_rel_path = line.strip()
            full_path = osp.join(self.image_dir, img_rel_path + '.jpg')
            pid = img_rel_path.split('/')[0]
            pid = pid2label[pid]  # Always convert to int label
            dataset.append((full_path, pid, camid))

        return dataset
