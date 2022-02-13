import os
import sys
import time
import torch
import random
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
from models.utils import Timer, logging, end_epoch_log, demo
from optimizers.warmup import WarmupCosine


class GAN(nn.Module):
    """GAN model used for Pix2Pix tasks
    Adapted from https://github.com/phillipi/pix2pix
    """
    def __init__(self, encoder, decoder, discriminator):
        super(GAN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def prepare_evaluate(self, cfg):
        self.timer = Timer(0, cfg['num_epochs'])
        self.recorder = RecorderMeter(cfg['metric_content'][:-1])

        self.l1_weight = cfg['l_norm_weight']
        self.gan_loss_weight = cfg['gan_loss_weight']
        self.criterionL1 = cfg['l1_loss'].to(self.device_list[0])
        self.criterionGAN = cfg['gan_criterion'].to(self.device_list[0])
        self.optimizer_G = cfg['g_optimizer']
        self.optimizer_D = cfg['d_optimizer']
        self.scheduler_G = cfg['g_lr_scheduler'] if 'g_lr_scheduler' in cfg and cfg['g_lr_scheduler'] else None
        self.scheduler_D = cfg['d_lr_scheduler'] if 'd_lr_scheduler' in cfg and cfg['d_lr_scheduler'] else None

    def evaluate_for_seed(self, cfg, train_loader, valid_loader, test_loader, logger, model_db):
        self.prepare_evaluate(cfg)

        for epoch in range(cfg['num_epochs']):
            # Train
            g_loss, d_loss, train_l1_loss, train_ssim = self.procedure(epoch, cfg, train_loader, logger, self.timer,
                                                                       'train')

            # Validation
            with torch.no_grad():
                val_l1_loss, val_ssim = self.procedure(epoch, cfg, valid_loader, logger, self.timer, 'val')

            # Test
            with torch.no_grad():
                test_l1_loss, test_ssim = self.procedure(epoch, cfg, test_loader, logger, self.timer, 'test')

            if self.rank == 0 or len(self.device_list) == 1:
                metrics = {
                    'train_g_loss': g_loss.item(),
                    'train_d_loss': d_loss.item(),
                    'train_l1_loss': train_l1_loss.item(),
                    'train_ssim': train_ssim.item(),
                    'val_l1_loss': val_l1_loss.item(),
                    'val_ssim': val_ssim.item(),
                    'test_l1_loss': test_l1_loss.item(),
                    'test_ssim': test_ssim.item(),
                    'time_elapsed': self.timer.elapse_string()
                }

                model_dic = {'encoder': self.encoder, 'decoder': self.decoder, 'discriminator': self.discriminator}
                end_epoch_log(cfg, epoch, metrics, model_dic, model_db, logger, self.recorder, self.timer)

    def procedure(self, epoch, cfg, data_loader, logger, timer, mode):
        self.train(True if mode == 'train' else False)

        train_g_meter, train_d_meter = AverageMeter(), AverageMeter()
        l1_meter, ssim_meter = AverageMeter(), AverageMeter()

        dataiter = iter(data_loader)
        for step in range(1, len(data_loader) + 1):

            # 1. setup input
            batch = next(dataiter)
            self.real_A = batch['image'].to(self.rank)
            self.real_B = batch['label'].to(self.rank)

            # 2. generate fake_B: G(A)
            self.fake_B = self.forward(self.real_A)

            # 3. update D
            self.set_requires_grad(self.discriminator, True)  # enable backprop for D
            self.optimizer_D.zero_grad()
            self.update_D(epoch, step, len(data_loader), mode)

            # 4. update G
            self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()
            self.update_G(epoch, step, len(data_loader), mode)

            # 5. demo & recording
            if (epoch == 0 and step == 1) or (step == len(data_loader) and mode == 'val'):
                demo(cfg, epoch, step, self.real_A[:3], self.real_B[:3], self.fake_B[:3], extra_msg='')
            self.ssim = cfg['ssim'](self.real_B, self.fake_B.clone().float().detach())
            train_d_meter.update(self.loss_D)
            train_g_meter.update(self.loss_G)
            l1_meter.update(self.loss_G_L1)
            ssim_meter.update(self.ssim)

            if mode == 'train' and (step == 1 or step % 200 == 0):
                metrics = {
                    'g_loss': self.loss_G.item(),
                    'd_loss': self.loss_D.item(),
                    'l1_loss': self.loss_G_L1.item(),
                    'ssim': self.ssim.item(),
                }
                logging(cfg, epoch, metrics, logger, timer, training=True, step=step, total_step=len(data_loader))

        return [train_g_meter.avg, train_d_meter.avg, l1_meter.avg, ssim_meter.avg] if mode == 'train' else \
            [l1_meter.avg, ssim_meter.avg]

    def update_D(self, epoch, step, total_step, mode):
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs
        D_pred_fake = self.discriminator(fake_AB.detach())
        D_label_fake = torch.tensor(0.).expand_as(D_pred_fake).to(self.rank)
        self.loss_D_fake = self.criterionGAN(D_pred_fake, D_label_fake)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        D_pred_real = self.discriminator(real_AB)
        D_label_real = torch.tensor(1.).expand_as(D_pred_real).to(self.rank)
        self.loss_D_real = self.criterionGAN(D_pred_real, D_label_real)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # optimization
        if mode == 'train':
            self.loss_D.backward()
            self.optimizer_D.step()
            if self.scheduler_D:
                assert isinstance(self.scheduler_D, WarmupCosine)
                self.scheduler_D.step(epoch + step / total_step)

    def update_G(self, epoch, step, total_step, mode):
        # G should try to generate real image, fool D
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        D_pred_fake = self.discriminator(fake_AB)
        G_label_real = torch.tensor(1.).expand_as(D_pred_fake).to(self.rank)
        self.loss_G_GAN = self.criterionGAN(D_pred_fake, G_label_real) * self.gan_loss_weight  # aim at D(G(A)) = 1.
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.l1_weight  # aim at G(A) = B
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # optimization
        if mode == 'train':
            self.loss_G.backward()
            self.optimizer_G.step()
            if self.scheduler_G:
                assert isinstance(self.scheduler_G, WarmupCosine)
                self.scheduler_G.step(epoch + step / total_step)

    def to_device(self, device_list, rank=None, ddp=False):
        self.device_list = device_list
        if len(self.device_list) > 1:
            if ddp:
                self.encoder = NaiveSyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.decoder = NaiveSyncBatchNorm.convert_sync_batchnorm(self.decoder)
                self.discriminator = NaiveSyncBatchNorm.convert_sync_batchnorm(self.discriminator)
                self.encoder = DDP(self.encoder.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.decoder = DDP(self.decoder.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.discriminator = DDP(self.discriminator.to(rank), device_ids=[rank], find_unused_parameters=True)
                self.rank = rank
            else:
                self.encoder = nn.DataParallel(self.encoder).to(self.device_list[0])
                self.decoder = nn.DataParallel(self.decoder).to(self.device_list[0])
                self.discriminator = nn.DataParallel(self.discriminator).to(self.device_list[0])
                self.rank = rank
        else:
            self.rank = self.device_list[0]
            self.to(self.rank)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def denormalize(imgs, mean, std):
        for i, (m, s) in enumerate(zip(mean, std)):
            imgs[:, i, :, :] = imgs[:, i, :, :] * s + m
        return imgs
