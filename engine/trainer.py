# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import time
import os.path as osp

import torch
import torch.nn as nn
import huepy as hue
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP
from .logger import MetricLogger


from test_person_search.libs.datasets import get_data_loader
from test_person_search.libs.utils.evaluator import GT_inference, inference, detection_performance_calc
import shutil
# from .osutils import mkdir_if_missing


global ITER
ITER = 0

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    # mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


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

    model.cuda()

    best_mAP = 0
    for epoch in range(epochs+1):
        metric_logger = MetricLogger()
        print(hue.info(hue.bold(hue.green("Start training from %s epoch"%str(epoch)))))
        start_time = time.time()
        model.train()
        scheduler.step()
        
        for iteration, data in enumerate(train_loader):
            steps = epoch*len(train_loader) + iteration
            if steps % log_period == 0:
                start = time.time()

            optimizer.zero_grad()
            img, target = data
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            score, feat = model(img)
            loss = loss_fn(score, feat, target)
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
    



def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader, val_loader,
        gallery_loader, probe_loader,
        optimizer,
        optimizer_center,
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
    # trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)

    cetner_loss_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT
    model.cuda()

    best_mAP = 0
    for epoch in range(epochs+1):
        metric_logger = MetricLogger()
        print(hue.info(hue.bold(hue.green("Start training from %s epoch"%str(epoch)))))
        start_time = time.time()
        model.train()
        scheduler.step()
        
        for iteration, data in enumerate(train_loader):
            steps = epoch*len(train_loader) + iteration
            if steps % log_period == 0:
                start = time.time()

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img, target = data
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            score, feat = model(img)
            loss = loss_fn(score, feat, target)
            loss.backward()
            optimizer.step()
            
            for param in center_criterion.parameters():
                param.grad.data *= (1. / cetner_loss_weight)
            optimizer_center.step()

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

        # scheduler.step()
        
        if (epoch % eval_period == 0) and (epoch!=0):
            print(hue.info(hue.bold(hue.lightgreen('Evaluate with Dukemtmc dataset'))))

            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            # save_checkpoint({
            #     'state_dict': model.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(output_dir,'checkpoint.pth.tar'))

            print(hue.info(hue.bold(hue.lightgreen('Working directory: {}'.format(output_dir)))))
            print(hue.run('Test with latest model:'))
            name_to_boxes, all_feats, probe_feats = GT_inference(model, gallery_loader, probe_loader, device)

            # name_to_boxes, all_feats, probe_feats = inference(model, gallery_loader, probe_loader, device, dataset_target)
            print(hue.run('Evaluating GT detections:'))
            precision, recall = detection_performance_calc(gallery_loader.dataset,
                                                        name_to_boxes.values(),
                                                        det_thresh=0.01)
            print(hue.run('Evaluating GT search:'))
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
