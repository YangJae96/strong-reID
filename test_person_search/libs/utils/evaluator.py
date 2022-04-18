import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import average_precision_score, precision_recall_curve
from torchvision.utils import save_image
from PIL import Image


# from ice.utils.data import transforms as T

import torchvision.transforms as T



import os.path as osp

import torch
import torch.nn.functional as F
from .misc import ship_data_to_cuda
import json
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([
        T.Resize((256,128), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])


## Detect
@torch.no_grad()
def inference(model, gallery_loader, probe_loader, device, dataset):
    cpu = torch.device('cpu')
    model.eval()

    if dataset=='prw':
        dataset ='prw_det'
    elif dataset=='cuhk':
        dataset ='cuhk_det'
    
    # if dataset=='cuhk2prw':
    #     dataset ='prw_det'
    # elif dataset=='prw2cuhk':
    #     dataset ='cuhk_det'

    if dataset=='prw_det':
        # load_path = osp.join(args.det_dataset_path, 'source_det2prw_gallery.json')
        load_path = '/root/workplace/CUCPS_Renewed/logs/det_cuhk/Mar25_22-30-15/det2prw_train/source_det2prw_gallery.json'
        with open(load_path, 'r') as fp:
            bbox_dict = json.load(fp)
    elif dataset=='cuhk_det':
        # load_path = osp.join(args.det_dataset_path, 'source_det2cuhk_gallery.json')
        load_path = '/root/workplace/CUCPS_Renewed/logs/det_prw/Mar25_22-32-54/det2cuhk_train/source_det2cuhk_gallery.json'
        with open(load_path, 'r') as fp:
            bbox_dict = json.load(fp)
    else:
        raise ValueError('Wrong dataset')


    im_names, all_boxes, all_labels, all_feats = [], [], [], []
    for data in tqdm(gallery_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, 'cuda')
        for im, t in zip(images, targets):
            
            pim_list = []
            ## No bbox detected in gallery img
            if len(bbox_dict[t['im_name']])==0:  
                all_feats.append(np.zeros((1,2048)))
                im_names.append(t['im_name'])
                all_boxes.append(np.zeros((1,5)))
            else:
                for bbox in bbox_dict[t['im_name']]:
                    x1, y1, x2, y2, score =  bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    pim = test_transformer(im.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
                    pim_list.append(pim)
                pims = torch.cat(pim_list, dim=0).cuda()
                outputs = model(pims).data
                all_feats.append(outputs.to(cpu).numpy())

                im_names.append(t['im_name'])

                bboxes = np.array(bbox_dict[t['im_name']])
                box_w_scores = torch.cat([torch.tensor(bboxes[:,:4]),
                                        torch.tensor(bboxes[:,4]).unsqueeze(1)],
                                        dim=1)
                all_boxes.append(box_w_scores.cpu().numpy())

            assert len(outputs) == len(box_w_scores)
            assert len(all_feats) == len(im_names)

    probe_feats = []
    for data in tqdm(probe_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, 'cuda')
        
        for im, t in zip(images, targets):
            
            pim_list = []
            for bbox in t['boxes']:
                x1, y1, x2, y2 =  bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                pim = test_transformer(im.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
                pim_list.append(pim)
            pims = torch.cat(pim_list, dim=0).cuda()
            outputs = model(pims).data
            probe_feats.append(outputs.to(cpu).numpy())

    name_to_boxes = OrderedDict(zip(im_names, all_boxes))

    return name_to_boxes, all_feats, probe_feats

## Detect
@torch.no_grad()
def inference_tSNE(model, gallery_loader, probe_loader, device, output_feature):
    cpu = torch.device('cpu')

    # with open('/root/workspace/Personsearch/datasets/PRW/gallery_bbox.json', 'r') as fp:
    with open('/root/workspace/Personsearch/datasets/CUHK/gallery_bbox.json', 'r') as fp:
        bbox_dict = json.load(fp)

    im_names, all_boxes, all_labels, all_feats = [], [], [], []
    for data in tqdm(gallery_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, 'cuda')
        for im, t in zip(images, targets):
            
            #     continue
            #     all_feats.append(np.zeros(outputs.shape))
            #     im_names.append(t['im_name'])
            #     all_boxes.append(np.zeros(box_w_scores.shape))
            #     for labels in t['labels']:
            #         all_labels.append(labels.cpu().numpy())
            # else:
            pim_list = []
            if len(bbox_dict[t['im_name']][0])!=0: 
                for bbox in bbox_dict[t['im_name']][0]:
                    x1, y1, x2, y2 =  bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    pim = test_transformer(im.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
                    pim_list.append(pim)
                    im_names.append(t['im_name'])

                pims = torch.cat(pim_list, dim=0).cuda()
                outputs = model(pims, output_feature).data
                all_feats.append(outputs.to(cpu).numpy())

                box_w_scores = torch.cat([torch.tensor(bbox_dict[t['im_name']][0]),
                                        torch.tensor(bbox_dict[t['im_name']][2]).unsqueeze(1)],
                                        dim=1)
                all_boxes.append(box_w_scores.cpu().numpy())
                all_labels.append(np.array(bbox_dict[t['im_name']][1]))
                
    probe_feats = []
    for data in tqdm(probe_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, 'cuda')
        
        for im, t in zip(images, targets):
            
            pim_list = []
            for bbox in t['boxes']:
                x1, y1, x2, y2 =  bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                pim = test_transformer(im.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
                pim_list.append(pim)
            pims = torch.cat(pim_list, dim=0).cuda()
            outputs = model(pims, output_feature).data
            probe_feats.append(outputs.to(cpu).numpy())

    return im_names, all_feats, all_labels, probe_feats
    # return name_to_boxes, all_feats, all_labels, probe_feats


# GT
@torch.no_grad()
def GT_inference(model, gallery_loader, probe_loader, device):
    cpu = torch.device('cpu')
    model.eval()
    im_names, all_boxes, all_feats = [], [], []
    for data in tqdm(gallery_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        for im, t in zip(images, targets):
            
            pim_list = []
            for bbox in t['boxes']:
                x1, y1, x2, y2 =  bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                pim = test_transformer(im.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
                pim_list.append(pim)
            pims = torch.cat(pim_list, dim=0).cuda()
            outputs = model(pims).data
            all_feats.append(outputs.to(cpu).numpy())

            im_names.append(t['im_name'])
            box_w_scores = torch.cat([t['boxes'],
                                    torch.ones((t['boxes'].shape[0],1)).cuda()],
                                    dim=1)
            all_boxes.append(box_w_scores.to(cpu).numpy())

    probe_feats = []
    for data in tqdm(probe_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        
        for im, t in zip(images, targets):
            
            pim_list = []
            for bbox in t['boxes']:
                x1, y1, x2, y2 =  bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                pim = test_transformer(im.crop((x1, y1, x2, y2))).unsqueeze(0).cuda()
                pim_list.append(pim)
            pims = torch.cat(pim_list, dim=0).cuda()
            outputs = model(pims).data
            probe_feats.append(outputs.to(cpu).numpy())
    
    name_to_boxes = OrderedDict(zip(im_names, all_boxes))

    return name_to_boxes, all_feats, probe_feats


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def detection_performance_calc(dataset, gallery_det, det_thresh=0.5, iou_thresh=0.5,
                               labeled_only=False):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image

    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert len(dataset) == len(gallery_det)
    gt_roidb = dataset.record

    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for gt, det in zip(gt_roidb, gallery_det):
        gt_boxes = gt['boxes']
        if labeled_only:
            inds = np.where(gt['gt_pids'].ravel() > 0)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        if det != []:
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
        else:
            num_det = 0
        if num_det == 0:
            count_gt += num_gt
            continue
        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
        tfmat = (ious >= iou_thresh)
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            if tfmat[:, j].any():
                y_true.append(True)
            else:
                y_true.append(False)
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / count_gt
    ap = average_precision_score(y_true, y_score) * det_rate
    precision, recall, __ = precision_recall_curve(y_true, y_score)
    recall *= det_rate

    print('{} detection:'.format('labeled only' if labeled_only else
                                 'all'))
    print('  recall = {:.2%}'.format(det_rate))
    if not labeled_only:
        print('  ap = {:.2%}'.format(ap))
    return ap, det_rate
