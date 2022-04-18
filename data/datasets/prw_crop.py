from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import shutil
import os
from .base_dataset import BaseImageDataset

class PRW_CROP(BaseImageDataset):
    """
    PRW
    Reference:
    PRW: A Benchmark. CVPR 2017.
    Dataset statistics:
    # identities: 932 (+1 for background)
    ## Train : 482 ID 
    # images: 18084 (train) + 3368 (query) + 15913 (gallery,test)
    
    # Scene images : 5704 (train), 6112 (test)
    # GT bbox : 18048 (train), 25062 (test)
    # 2057 (query) -> GT bbox 
    """
    dataset_dir = ''

    def __init__(self, root, verbose=True):
        super(PRW_CROP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.dataset_dir = osp.join(self.dataset_dir, 'prw')
        print("self.dataset_dir == ",self.dataset_dir)

        ### gt_bbox_cropped_train_unsupervised -> No junk image for training
        self.train_dir = osp.join(self.dataset_dir, 'gt_bbox_cropped_train')
        self.query_dir = osp.join(self.dataset_dir, 'query_box')
        self.gallery_dir = osp.join(self.dataset_dir, 'gt_bbox_cropped_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PRW loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        # self.clean_samples = clean_bbox

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            ## img_path = "440_c4s2_072373_4"
            ## pid = 440,  camid=4
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid <0: continue  # junk images are just ignored
            assert 0 <= pid <= 30000  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        # print("dataset == ",dataset)

        return dataset