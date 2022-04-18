from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob

import re
import shutil
import os
import os.path as osp
import glob
import re
import shutil
import os
from .base_dataset import BaseImageDataset

class CUHK_CROP(BaseImageDataset):
    """
    CUHK
    Reference:
    CUHK: A Benchmark. CVPR 2017.
    Dataset statistics:
    # identities: 5533 (+1 for background)
    # Scene images : 11206??(train), 6978 (test)
    # GT bbox : 55272 (train), 25062 (test)       labeled : 23,387 (train), 14,453 (test)
    # 2900 (query) -> GT bbox 
    """
    dataset_dir = ''
    def __init__(self, root, verbose=True):
        super(CUHK_CROP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.dataset_dir = osp.join(self.dataset_dir, 'CUHK-SYSU')
        print("self.dataset_dir == ",self.dataset_dir)

        ### gt_bbox_cropped_train_unsupervised -> No junk image for training
        self.train_dir = osp.join(self.dataset_dir, 'gt_bbox_cropped_train') ## 23,387
        self.query_dir = osp.join(self.dataset_dir, 'query_box') ## 2900
        self.gallery_dir = osp.join(self.dataset_dir, 'gt_bbox_cropped_test') ## 14453

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
        pattern = re.compile(r'([-\d]+)_s(\d)')
        # print("pattern == ",pattern)
# 
        pid_container = set()
        for img_path in img_paths:
            # print("pattern.search(img_path == ",pattern.search(img_path))
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            ## img_path = "440_c4s2_072373_4"
            ## pid = 440,  camid=4
            # print("pattern.search(img_path) == ",pattern.search(img_path))
            pid, _ = map(int, pattern.search(img_path).groups())
            camid = 0
            if pid <0: continue  # junk images are just ignored
            # print("pid == ",pid)
            # print("camid == ",camid)
            assert 0 <= pid <= 30000  # pid == 0 means background
            assert 0 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        # print("dataset == ",dataset)

        return dataset


# class CUHK_CROP(BaseImageDataset):
#     """
#     CUHK
#     Reference:
#     CUHK: A Benchmark. CVPR 2017.
#     Dataset statistics:
#     # identities: 5533 (+1 for background)
#     # Scene images : 11206??(train), 6978 (test)
#     # GT bbox : 55272 (train), 25062 (test)       labeled : 23,387 (train), 14,453 (test)
#     # 2900 (query) -> GT bbox 
#     """
#     dataset_dir = ''
#     def __init__(self, root, verbose=True):
#         super(CUHK_CROP, self).__init__()
#         self.dataset_dir = osp.join(root, self.dataset_dir)
# 
#         self.dataset_dir = osp.join(self.dataset_dir, 'CUHK-SYSU')
#         print("self.dataset_dir == ",self.dataset_dir)
# 
#         ### gt_bbox_cropped_train_unsupervised -> No junk image for training
#         self.train_dir = osp.join(self.dataset_dir, 'gt_bbox_cropped_train') ## 23,387
#         # self.query_dir = osp.join(self.dataset_dir, 'query_box') ## 2900
#         # self.gallery_dir = osp.join(self.dataset_dir, 'gt_bbox_cropped_test') ## 14453
# 
#         # train = self._process_dir(self.train_dir, relabel=True)
# 
#         self.train, self.query, self.gallery = [], [], []
#         self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
# 
#     def train_preprocess(self):
#         ret = []
#         fpaths = sorted(glob(osp.join(self.train_dir, '*.jpg')))
#         for fpath in fpaths:
#             fname = fpath.split('/')[-1]
#             cnt = fname[:-4].split('_')[1]
# 
#             if cnt < 0: continue
#             ret.append((fname, int(cnt), 1))
#         return ret, len(ret)
# 
#     def load(self, info=True):
#         self.train, self.num_train_ids = self.train_preprocess(self.images_dir)
# 
#         if info:
#             print(self.__class__.__name__, self.name, "loaded")
#             print("  subset   | # ids | # images")
#             print("  ---------------------------")
#             print("  train    | 'Unknown' | {:8d}"
#                 .format(len(self.train)))
#             print("  query    | {:5d} | {:8d}"
#                 .format(self.num_query_ids, len(self.query)))
#             print("  gallery  | {:5d} | {:8d}"
#                 .format(self.num_gallery_ids, len(self.gallery)))