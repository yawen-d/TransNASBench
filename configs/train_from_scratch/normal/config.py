import os
import sys
import torch
from pathlib import Path

sys.path.insert(1, str((Path(__file__).parent / '..' / '..' / '..' / 'lib').resolve()))

from models.gan import GAN
from models.decoder import GenerativeDecoder
from models.encoder import FFEncoder
from models.discriminator import Discriminator
import losses.all as loss_lib
import data.load_ops as load_ops
from optimizers.warmup import WarmupCosine
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def get_cfg(encoder_str):
    cfg = {}

    # basics
    cfg['encoder_str'] = encoder_str
    cfg['config_dir'] = str(Path(__file__).parent.resolve())
    cfg['task_name'] = Path(cfg['config_dir']).name

    # paths
    cfg['root_dir'] = str((Path(__file__).parent / '..' / '..' / '..' / '..' / '..').resolve())
    cfg['dataset_dir'] = str(Path(cfg['root_dir']) / 'data/taskonomy_data/taskonomydata_mini')
    cfg['data_split_dir'] = str(Path(cfg['root_dir']) / 'tb101/code/experiments/final5k')
    cfg['log_dir'] = str(
        Path(cfg['root_dir']) / "tb101/benchmark_results/benchmark_results_local" / cfg['task_name'] / 'model_results' / cfg['encoder_str'])

    cfg['s3_dir'] = ''  # s3://bucket-hang/cynthia/Result/Task/
    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'

    # data loading
    cfg['batch_size'] = 128
    cfg['num_workers'] = 8

    # inputs
    cfg['input_dim'] = (256, 256)  # (1024, 1024)
    cfg['input_num_channels'] = 3

    # targets
    cfg['target_dim'] = (256, 256)  # (1024, 1024)
    cfg['target_channel'] = 3
    cfg['target_load_fn'] = load_ops.load_raw_img_label
    cfg['target_load_kwargs'] = {}

    # demo
    cfg['demo_kwargs'] = {}

    # transform
    cfg['normal_params'] = {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    }
    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        # load_ops.RandomHorizontalFlip(0.5),
        # load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params'])
    ])

    # model
    cfg['encoder'] = FFEncoder(encoder_str, task_name=cfg['task_name']).network
    cfg['decoder_input_dim'] = (2048, 16, 16) if cfg['encoder_str'] == 'resnet50' else cfg['encoder'].output_dim
    cfg['decoder'] = GenerativeDecoder(cfg['decoder_input_dim'], cfg['target_dim'])
    cfg['discriminator'] = Discriminator()
    # cfg['discriminator'] = PatchGANDiscriminator(cfg['input_num_channels'] + cfg['target_channel'])
    cfg['model_type'] = GAN(cfg['encoder'], cfg['decoder'], cfg['discriminator'])

    # train
    cfg['fp16'] = False
    cfg['amp_opt_level'] = 'O1'
    cfg['num_epochs'] = 30
    cfg['l1_loss'] = torch.nn.L1Loss()
    cfg['ssim'] = SSIM(data_range=1, size_average=True, channel=3)
    cfg['l_norm_weight'] = 0.99
    cfg['gan_loss_weight'] = 1 - cfg['l_norm_weight']
    cfg['gan_criterion'] = loss_lib.GANLoss('vanilla')  # ['lsgan', 'vanilla']
    # cfg['g_optimizer'] = torch.optim.SGD
    # cfg['d_optimizer'] = torch.optim.SGD
    cfg['g_optimizer'] = torch.optim.Adam
    cfg['d_optimizer'] = torch.optim.Adam
    cfg['initial_lr'] = 0.0001
    cfg['d_lr'] = 0.0001
    cfg['g_optimizer_kwargs'] = {
        'lr': cfg['initial_lr'],
        'betas': (0.5, 0.999),
        'weight_decay': 0.0001
    }
    cfg['d_optimizer_kwargs'] = {
        'lr': cfg['d_lr'],
        'betas': (0.5, 0.999),
        'weight_decay': 0.0001
    }
    # cfg['g_lr_scheduler'] = WarmupCosine
    # cfg['d_lr_scheduler'] = WarmupCosine

    # cfg['identity_elems'] = {'seed': cfg['seed'], 'batch-size': cfg['batch_size'], 'lr': cfg['initial_lr']}
    cfg['metric_content'] = ['train_g_loss', 'train_d_loss', 'train_l1_loss', 'train_ssim',
                             'val_l1_loss', 'val_ssim', 'test_l1_loss', 'test_ssim',
                             'time_elapsed']
    cfg['plot_msg'] = f"lr_{cfg['initial_lr']}"

    return cfg
