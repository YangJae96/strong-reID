# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import time
import os.path as osp
import os 

import torch
import torch.nn as nn
import huepy as hue
from .logger import MetricLogger
from test_person_search.libs.datasets import get_data_loader
from test_person_search.libs.utils.evaluator import GT_inference, inference, detection_performance_calc
import shutil

global ITER
ITER = 0

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    # mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))

def do_train(
        cfg,
        model,
        train_loader,
        gallery_loader, probe_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        dataset_target
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    print(hue.info(hue.bold(hue.lightgreen('Working directory: {}'.format(cfg.OUTPUT_DIR)))))

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model.cuda()

    best_mAP = 0
    for epoch in range(epochs+1):
        metric_logger = MetricLogger()
        print(hue.info(hue.bold(hue.green("Start training from %s epoch"%str(epoch)))))
        start_time = time.time()
        model.train()

        for iteration, data in enumerate(train_loader):
            steps = epoch*len(train_loader) + iteration
            if steps % log_period == 0:
                start = time.time()

            optimizer.zero_grad()
            img1, img2, target, _, _, = data

            # print("img1 == ",img1.shape) ## (64,3,256,128)

            images = torch.cat([img1, img2], dim=0) ## Original and augmented view
            # print("images == ",images.shape) ## (128,3,256,128)


            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            bsz = target.shape[0]

            # images = images.to(device) if torch.cuda.device_count() >= 1 else images
            # target = target.to(device) if torch.cuda.device_count() >= 1 else target
            score, feat = model(images)

            # print("score == ",score.shape)
            # print("feat == ",feat.shape)
            # print("target == ",target.shape)

            # # compute loss
            # features = model(images)
            f1, f2 = torch.split(feat, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) ## (256,2,128)
            # # print("labels == ",labels.shape) ## (256)
            # if opt.method == 'SupCon':
            #     loss = criterion(features, labels)
            loss = loss_fn(score[:bsz], features, target)

            loss.backward()
            optimizer.step()
            
            if steps % log_period == 0:
                # Print 
                loss_value = loss.item()
                state = dict(loss_value=loss_value,
                            lr=scheduler.get_lr()[0])
                # Update logger
                batch_time = time.time() - start
                metric_logger.update(batch_time=batch_time)
                metric_logger.update(**state)
                    
                # Print log on console
                metric_logger.print_log(epoch, iteration, len(train_loader))
            else:
                state = None
        scheduler.step()
        
        if (epoch % eval_period == 0) and (epoch!=0):

            print(hue.info(hue.bold(hue.lightgreen('Working directory: {}'.format(output_dir)))))
            print(hue.run('Test with latest model:'))
            name_to_boxes, all_feats, probe_feats = GT_inference(model, gallery_loader, probe_loader, device)

            # name_to_boxes, all_feats, probe_feats = inference(model, gallery_loader, probe_loader, device, dataset_target)
            print(hue.run('Evaluating detections:'))
            precision, recall = detection_performance_calc(gallery_loader.dataset,
                                                        name_to_boxes.values(),
                                                        det_thresh=0.01)
            print(hue.run('Evaluating search:'))
            # gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
            gallery_size = 100 if dataset_target == 'cuhk' else -1
            ret = gallery_loader.dataset.search_performance_calc(
                gallery_loader.dataset, probe_loader.dataset,
                name_to_boxes.values(), all_feats, probe_feats,
                det_thresh=0.9, gallery_size=gallery_size,
                ignore_cam_id=True,
                remove_unlabel=True)

            mAP_1 = ret['mAP']
            is_best = mAP_1 > best_mAP
            best_mAP = max(mAP_1, best_mAP)

            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_mAP': best_mAP,
            }
            
            fpath = osp.join(output_dir, str(epoch)+ '_checkpoint.pth.tar')
            torch.save(state, fpath)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(output_dir,'checkpoint.pth.tar'))
    
    # trainer.run(train_loader, max_epochs=epochs)
# 
#     epoch = 1
    # evaluator.run(val_loader)
    # cmc, mAP = evaluator.state.metrics['r1_mAP']
    # logger.info("Validation Results - Epoch: {}".format(epoch))
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/market1501/Experiment-all-tricks-tri_center-256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005_middle')"
