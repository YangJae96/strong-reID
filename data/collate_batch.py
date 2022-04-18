# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    img1, img2, pids, _, _, = zip(*batch)

    # return img1, img2, pid, camid, img_path
    pids = torch.tensor(pids, dtype=torch.int64)
    # print("img1[0] == ",img1[0].shape) # (3,256,128)
    # images = torch.cat([img1[0], img2[0]], dim=0) ## Original and augmented view
    # print("images == ",images.shape)
    # return torch.stack(images, dim=0), pids
    return img1, img2, pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
