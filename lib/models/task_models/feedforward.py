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


class FeedForwardNet(nn.Module):
    """FeedForwardNet class used by classification and regression tasks"""

    def __init__(self, encoder, decoder):
        super(FeedForwardNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

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
            imgs = batch['image'].to(self.rank)
            labels = batch['label'].to(self.rank)

            # 2. forward
            logits = self.forward(imgs)

            # 3. metrics
            loss = cfg['criterion'](logits, labels)
            top1, top5 = get_topk_acc(logits, labels.argmax(dim=-1), topk=(1, 5))

            # 4. optimize parameters
            if mode == 'train':
                cfg['optimizer'].zero_grad()
                loss.backward()
                cfg['optimizer'].step()

                if 'lr_scheduler' in cfg and cfg['lr_scheduler']:
                    cfg['lr_scheduler'].step(epoch + step / len(data_loader))

            # 5. demo & recording
            if (epoch == 0 and step == 1) or (step == len(data_loader) and mode == 'val'):
                demo(cfg, epoch, step, imgs[:3], labels[:3], logits[:3], extra_msg='')
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

    def evaluate_for_seed_darts(self, cfg, w_loader, a_loader, valid_loader, test_loader, logger, model_db):
        timer = Timer(0, cfg['num_epochs'])
        recorder = RecorderMeter(cfg['metric_content'][:-1])

        for epoch in range(cfg['num_epochs']):
            # Train
            w_train_top1, w_train_top5, w_train_loss, a_train_top1, a_train_top5, a_train_loss = \
                self.procedure_darts(epoch, cfg, w_loader, a_loader, logger, timer, 'train')

            # Validation
            with torch.no_grad():
                valid_top1, valid_top5, valid_loss = self.procedure(epoch, cfg, valid_loader, logger, timer, 'val')

            # Test
            with torch.no_grad():
                test_top1, test_top5, test_loss = self.procedure(epoch, cfg, test_loader, logger, timer, 'test')

            if self.rank == 0 or len(self.device_list) == 1:

                metrics = {
                    'w_train_loss': w_train_loss.item(),
                    'a_train_loss': a_train_loss.item(),
                    'valid_loss': valid_loss.item(),
                    'test_loss': test_loss.item(), }
                if cfg['task_name'] != 'room_layout':
                    metrics['w_train_top1'] = w_train_top1.tolist()[0]
                    metrics['w_train_top5'] = w_train_top5.tolist()[0]
                    metrics['a_train_top1'] = a_train_top1.tolist()[0]
                    metrics['a_train_top5'] = a_train_top5.tolist()[0]
                    metrics['valid_top1'] = valid_top1.tolist()[0]
                    metrics['valid_top5'] = valid_top5.tolist()[0]
                    metrics['test_top1'] = test_top1.tolist()[0]
                    metrics['test_top5'] = test_top5.tolist()[0]
                metrics['time_elapsed'] = timer.elapse_string()

                model_dic = {'encoder': self.encoder, 'decoder': self.decoder}
                end_epoch_log(cfg, epoch, metrics, model_dic, model_db, logger, recorder, timer)

                Sstr = f'*SEARCH* [{epoch}][{timer.elapse_string()}]\n'
                Wstr = f'Weights [Loss {w_train_loss:.3f}  Prec@1 {w_train_top1:.2f} Prec@5 {w_train_top5:.2f}]\n'
                Astr = f'Arch [Loss {a_train_loss:.3f}  Prec@1 {a_train_top1:.2f} Prec@5 {a_train_top5:.2f}]\n'
                alpha = self.encoder.get_alphas()[0]
                arch = alpha.argmax(dim=1).tolist()
                arch2list = lambda l, encoder_str: encoder_str.split('-')[:2] + [f"{l[0]}_{l[1]}{l[2]}_{l[3]}{l[4]}{l[5]}"]
                Astr += f"Current Alphas: {alpha};\n Early Stopped Arch: {'-'.join(arch2list(arch, cfg['encoder_str']))}"

                logger.write(Sstr + Wstr + Astr)


    def procedure_darts(self, epoch, cfg, w_loader, a_loader, logger, timer, mode):
        assert mode == 'train'
        self.train(True if mode == 'train' else False)
        w_loss_meter, w_top1_meter, w_top5_meter = AverageMeter(), AverageMeter(), AverageMeter()
        a_loss_meter, a_top1_meter, a_top5_meter = AverageMeter(), AverageMeter(), AverageMeter()

        w_dataiter, a_dataiter = iter(w_loader), iter(a_loader)
        assert len(w_loader) == len(a_loader)
        for step in range(1, len(w_loader) + 1):

            # 1. setup input
            w_batch = next(w_dataiter)
            w_imgs = w_batch['image'].to(self.rank)
            w_labels = w_batch['label'].to(self.rank)
            a_batch = next(a_dataiter)
            a_imgs = a_batch['image'].to(self.rank)
            a_labels = a_batch['label'].to(self.rank)

            # 2. forward, metrics, optimization, record
            cfg['w_optimizer'].zero_grad()
            w_logits = self.forward(w_imgs)
            w_loss = cfg['criterion'](w_logits, w_labels)
            w_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            cfg['w_optimizer'].step()
            w_top1, w_top5 = get_topk_acc(w_logits, w_labels.argmax(dim=-1), topk=(1, 5))
            if 'w_lr_scheduler' in cfg and cfg['w_lr_scheduler']:
                cfg['w_lr_scheduler'].step(epoch + step / len(w_loader))

            cfg['a_optimizer'].zero_grad()
            a_logits = self.forward(a_imgs)
            a_loss = cfg['criterion'](a_logits, a_labels)
            a_loss.backward()
            cfg['a_optimizer'].step()
            a_top1, a_top5 = get_topk_acc(a_logits, a_labels.argmax(dim=-1), topk=(1, 5))
            if 'a_lr_scheduler' in cfg and cfg['a_lr_scheduler']:
                cfg['a_lr_scheduler'].step(epoch + step / len(w_loader))

            # 5. demo & recording
            if epoch == 0 and step == 1:
                demo(cfg, epoch, step, w_imgs[:3], w_labels[:3], w_logits[:3], extra_msg='')
            w_loss_meter.update(w_loss); a_loss_meter.update(a_loss)
            w_top1_meter.update(w_top1); a_top1_meter.update(a_top1)
            w_top5_meter.update(w_top5); a_top5_meter.update(a_top5)

            if step == 1 or step % 200 == 0:
                if cfg['task_name'] == 'room_layout':
                    metrics = {'w_loss': w_loss_meter.avg, 'a_loss': a_loss_meter.avg}
                else:
                    metrics = {
                        'w_loss': w_loss_meter.avg, 'a_loss': a_loss_meter.avg,
                        'w_train_top1': w_top1_meter.avg, 'a_train_top1': a_top1_meter.avg,
                        'w_train_top5': w_top5_meter.avg, 'a_train_top5': a_top5_meter.avg
                    }

                alpha = self.encoder.get_alphas()[0]
                arch = alpha.argmax(dim=1).tolist()
                arch2list = lambda l, encoder_str: encoder_str.split('-')[:2] + [
                    f"{l[0]}_{l[1]}{l[2]}_{l[3]}{l[4]}{l[5]}"]
                Astr = f"\nCurrent Alphas: {alpha};\n Early Stopped Arch: {'-'.join(arch2list(arch, cfg['encoder_str']))}"
                logging(cfg, epoch, metrics, logger, timer, training=True, step=step, total_step=len(w_loader),
                        extra_message=Astr)
        return [w_top1_meter.avg, w_top5_meter.avg, w_loss_meter.avg,
                a_top1_meter.avg, a_top5_meter.avg, a_loss_meter.avg]

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
