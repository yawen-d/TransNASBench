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
from models.net_ops.norm import NaiveSyncBatchNorm
from models.utils import Timer, get_topk_acc, logging, end_epoch_log, demo


class SiameseNet(nn.Module):
    """SiameseNet used in Jigsaw task"""
    def __init__(self, encoder, decoder):
        super(SiameseNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 4:
            assert x.shape == (1, 3, 720, 1080)
            x = image2tiles4testing(x)
        imgtile_num = x.shape[1]
        encoder_output = []
        for index in range(imgtile_num):
            input_i = x[:, index, :, :, :]
            ith_encoder_output = self.encoder(input_i)
            encoder_output.append(ith_encoder_output)
        concat_output = torch.cat(encoder_output, axis=1)
        final_output = self.decoder(concat_output)
        return final_output

    def evaluate_for_seed(self, cfg, train_loader, valid_loader, test_loader, logger, model_db):
        timer = Timer(0, cfg['num_epochs'])
        recorder = RecorderMeter(cfg['metric_content'][:-1])

        for epoch in range(cfg['num_epochs']):
            # Train
            train_top1, train_top5, train_loss = self.procedure(epoch, cfg, train_loader, logger, timer, 'train')

            # Validation
            with torch.no_grad():
                valid_top1, valid_top5, valid_loss = self.procedure(epoch, cfg, valid_loader, logger, timer, 'val')

            # Test
            with torch.no_grad():
                test_top1, test_top5, test_loss = self.procedure(epoch, cfg, test_loader, logger, timer, 'test')

            if self.rank == 0 or len(self.device_list) == 1:
                metrics = {
                    'train_loss': train_loss.item(),
                    'valid_loss': valid_loss.item(),
                    'test_loss': test_loss.item(), }
                if cfg['task_name'] != 'room_layout':
                    metrics['train_top1'] = train_top1.tolist()[0]
                    metrics['train_top5'] = train_top5.tolist()[0]
                    metrics['valid_top1'] = valid_top1.tolist()[0]
                    metrics['valid_top5'] = valid_top5.tolist()[0]
                    metrics['test_top1'] = test_top1.tolist()[0]
                    metrics['test_top5'] = test_top5.tolist()[0]
                metrics['time_elapsed'] = timer.elapse_string()

                model_dic = {'encoder': self.encoder, 'decoder': self.decoder}
                end_epoch_log(cfg, epoch, metrics, model_dic, model_db, logger, recorder, timer)

    def procedure(self, epoch, cfg, data_loader, logger, timer, mode):
        self.train(True if mode == 'train' else False)
        loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()

        dataiter = iter(data_loader)
        for step in range(1, len(data_loader) + 1):

            # 1. setup input
            batch = next(dataiter)
            raws = batch['raw'].to(self.rank)
            imgs = batch['image'].to(self.rank)
            labels = batch['label'].to(self.rank)

            # 2. forward
            logits = self.forward(imgs)

            # 3. metrics
            loss = cfg['criterion'](logits, labels)
            top1, top5 = get_topk_acc(logits, labels, topk=(1, 5))

            # 4. optimize parameters
            if mode == 'train':
                cfg['optimizer'].zero_grad()
                loss.backward()

                cfg['optimizer'].step()

                if 'lr_scheduler' in cfg and cfg['lr_scheduler']:
                    cfg['lr_scheduler'].step(epoch + step / len(data_loader))

            # 5. demo & recording
            if (epoch == 0 and step == 1) or (step == len(data_loader) and mode == 'val'):
                demo(cfg, epoch, step, raws[:3], labels[:3], logits[:3], extra_msg='')
            loss_meter.update(loss)
            top1_meter.update(top1)
            top5_meter.update(top5)

            if mode == 'train' and (step == 1 or step % 200 == 0):
                if cfg['task_name'] == 'room_layout':
                    metrics = {'loss': loss_meter.avg}
                else:
                    metrics = {
                        'loss': loss_meter.avg,
                        'train_top1': top1_meter.avg,
                        'train_top5': top5_meter.avg
                    }
                logging(cfg, epoch, metrics, logger, timer, training=True, step=step, total_step=len(data_loader))
        return [top1_meter.avg, top5_meter.avg, loss_meter.avg]

    def to_device(self, device_list, rank=None, ddp=False):
        self.device_list = device_list
        if len(self.device_list) > 1:
            if ddp:
                self.encoder = NaiveSyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.decoder = NaiveSyncBatchNorm.convert_sync_batchnorm(self.decoder)
                self.encoder = DDP(self.encoder.to(rank), device_ids=[rank])
                self.decoder = DDP(self.decoder.to(rank), device_ids=[rank])
                self.rank = rank
            else:
                self.encoder = nn.DataParallel(self.encoder).to(self.device_list[0])
                self.decoder = nn.DataParallel(self.decoder).to(self.device_list[0])
                self.rank = rank
        else:
            self.rank = self.device_list[0]
            self.to(self.rank)


def image2tiles4testing(img, num_pieces=9):
    """
    Generate the 9 pieces input for Jigsaw task.

    Parameters:
    -----------
        img (tensor): Image to be cropped (1, 3, 720, 1080)
h
    Return:
    -----------
        img_tiles: tensor (1, 9, 3, 240, 360)
    """

    if num_pieces != 9:
        raise ValueError(f'Target permutation of Jigsaw is supposed to have length 9, getting {num_pieces} here')

    Ba, Ch, He, Wi = img.shape  # (1, 3, 720, 1080)

    unitH = int(He / 3)  # 240
    unitW = int(Wi / 3)  # 360

    return img.view(Ba, 9, Ch, unitH, unitW)
