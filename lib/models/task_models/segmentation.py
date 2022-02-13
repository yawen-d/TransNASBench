import os
import sys
import time
import torch
import random
import numpy as np
from torch import nn
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from data import load_ops
from log_utils import AverageMeter, RecorderMeter
from models.utils import Timer, get_topk_acc, get_iou, get_confusion_matrix, logging, end_epoch_log, demo
from optimizers.warmup import WarmupCosine


class Segmentation(nn.Module):
    """Segmentation used by semanticsegment task"""
    def __init__(self, encoder, decoder):
        super(Segmentation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def evaluate_for_seed(self, cfg, train_loader, valid_loader, test_loader, logger, model_db):
        self.timer = Timer(0, cfg['num_epochs'])
        self.recorder = RecorderMeter(cfg['metric_content'][:-1])

        for epoch in range(cfg['num_epochs']):
            # Train
            train_loss, train_acc, train_miou = self.procedure(epoch, cfg, train_loader, logger, self.timer, 'train')

            # Validation
            with torch.no_grad():
                val_loss, val_acc, val_miou = self.procedure(epoch, cfg, valid_loader, logger, self.timer, 'val')

            # Test
            with torch.no_grad():
                test_loss, test_acc, test_miou = self.procedure(epoch, cfg, test_loader, logger, self.timer, 'test')

            if self.rank == 0 or len(self.device_list) == 1:
                # Store Metrics
                metrics = {'train_loss': train_loss.item(), 'train_acc': train_acc.item(),
                           'train_mIoU': train_miou.item(), 'valid_loss': val_loss.item(), 'valid_acc': val_acc.item(),
                           'valid_mIoU': val_miou.item(), 'test_loss': test_loss.item(), 'test_acc': test_acc.item(),
                           'test_mIoU': test_miou.item(), 'time_elapsed': self.timer.elapse_string()}

                model_dic = {'encoder': self.encoder, 'decoder': self.decoder}
                end_epoch_log(cfg, epoch, metrics, model_dic, model_db, logger, self.recorder, self.timer)

    def procedure(self, epoch, cfg, data_loader, logger, timer, mode):
        self.train(True if mode == 'train' else False)
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        confusion_matrix = np.zeros((cfg['target_num_channel'], cfg['target_num_channel']))

        dataiter = iter(data_loader)
        for step in range(1, len(data_loader) + 1):

            # 1. setup input
            batch = next(dataiter)
            imgs = batch['image'].to(self.rank)
            labels = batch['label'].long().to(self.rank)

            # 2. forward
            outputs = self.forward(imgs)

            # 3. metrics
            _, preds = outputs.topk(1, 1, True, True)

            batch_loss = cfg['criterion'](outputs, labels)
            batch_acc = get_topk_acc(outputs, labels, topk=(1,))[0]
            batch_confusion_matrix = get_confusion_matrix(labels, preds, cfg['target_num_channel'])

            # 4. optimize parameters
            if mode == 'train':
                cfg['optimizer'].zero_grad()
                batch_loss.backward()
                cfg['optimizer'].step()

                if 'lr_scheduler' in cfg and cfg['lr_scheduler']:
                    if isinstance(cfg['lr_scheduler'], WarmupCosine):
                        cfg['lr_scheduler'].step(epoch + step / len(data_loader))
                    else:
                        if step == len(data_loader):
                            cfg['lr_scheduler'].step()

            # 5. demo & recording
            if (epoch == 0 and step == 1) or (step == len(data_loader) and mode == 'val'):
                demo(cfg, epoch, step, imgs[:3], labels[:3], preds[:3], extra_msg='')
            loss_meter.update(batch_loss)
            acc_meter.update(batch_acc)
            confusion_matrix += batch_confusion_matrix

            if mode == 'train' and (step == 1 or step % 200 == 0):
                _, running_miou = get_iou(confusion_matrix)
                metrics = {
                    'loss': loss_meter.avg,
                    'accuracy': acc_meter.avg,
                    'mIoU': running_miou
                }
                logging(cfg, epoch, metrics, logger, timer, training=True, step=step, total_step=len(data_loader))

        epoch_ious, epoch_miou = get_iou(confusion_matrix)
        return [loss_meter.avg, acc_meter.avg, epoch_miou]

    def to_device(self, device_list, rank=None, ddp=False):
        self.device_list = device_list
        if len(self.device_list) > 1:
            if ddp:
                self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder)
                self.encoder = DDP(self.encoder.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.decoder = DDP(self.decoder.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.rank = rank
            else:
                self.encoder = nn.DataParallel(self.encoder).to(self.device_list[0])
                self.decoder = nn.DataParallel(self.decoder).to(self.device_list[0])
                self.rank = rank
        else:
            self.rank = self.device_list[0]
            self.to(self.rank)
