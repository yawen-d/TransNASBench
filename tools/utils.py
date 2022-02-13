# utils.py defines helper functions for main.py, get_model_info.py
import os
import sys
import PIL
import torch
import random
import numpy as np
from torch import nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch.distributed as dist  # DDP

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
import data.load_ops as load_ops

sys.path.remove(str(lib_dir))


#######
# DDP #
#######

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'10003'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


################
# Preparations #
################

def prepare_seed_cudnn(seed, logger=None):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if logger:
        logger.write("--------------- {0:12}: {1} setup finished! ---------------".format("seed", seed))


def get_machine_info():
    info = "Python  Version  : {:}".format(sys.version.replace('\n', ' '))
    info += "\nPillow  Version  : {:}".format(PIL.__version__)
    info += "\nPyTorch Version  : {:}".format(torch.__version__)
    info += "\ncuDNN   Version  : {:}".format(torch.backends.cudnn.version())
    info += "\nCUDA available   : {:}".format(torch.cuda.is_available())
    info += "\nCUDA GPU numbers : {:}".format(torch.cuda.device_count())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        info += "\nCUDA_VISIBLE_DEVICES={:}".format(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        info += "\nDoes not set CUDA_VISIBLE_DEVICES"
    return info


def setup_config(args, world_size):
    """
    Raises: FileNotFoundError if 'config.py' doesn't exist in cfg_dir
    """
    print('cfg_dir', args.cfg_dir)
    if not (Path(args.cfg_dir) / 'config.py').is_file():
        raise ImportError('config.py not found in {0}'.format(args.cfg_dir))
    sys.path.insert(0, args.cfg_dir)
    from config import get_cfg
    cfg = get_cfg(args.encoder_str)
    # cleanup
    try:
        del sys.modules['config']
    except:
        pass
    sys.path.remove(args.cfg_dir)
    cfg['world_size'] = world_size
    # sys.exit()
    return cfg


def add_algo_cfg(cfg, args):
    # If trying to run benchmark algorithms, add relevant algo configs
    if 'benchmark_algo' in args.__dict__.keys():
        result_folder = f"{args.algo_output_dir}/{args.benchmark_algo}/{cfg['task_name']}/{args.seed}"
        cfg['log_dir'] = str(Path(cfg['root_dir']) / result_folder)
        if not (Path(args.algo_cfg_dir) / 'config.py').is_file():
            raise ImportError('config.py not found in {0}'.format(args.algo_cfg_dir))
        sys.path.insert(0, args.algo_cfg_dir)
        from config import get_cfg
        cfg.update(get_cfg(cfg))

        # cleanup
        try:
            del sys.modules['config']
        except:
            pass
        sys.path.remove(args.algo_cfg_dir)

        for arg in vars(args):
            if (arg in cfg and arg) or arg == 'seed':
                print(f"Set {arg} in config as {getattr(args, arg)}.")
                cfg[arg] = getattr(args, arg)
    return cfg


def mkdir_op(*paths):
    """
    make directories
    Args:
        *paths: the paths of dirs to make
    """
    for path in paths:
        os.system("mkdir -p {dir}".format(dir=path))


def setup_model(cfg, device_list, rank=None, ddp=False):
    """
    Setup model
    Args:
        cfg: cfg dict directly loaded from config.py
        device_list: device list available
        rank: current world for ddp
        ddp: whether training with ddp

    Returns:
        the already set up model
    """
    # to device or DistributedDataParallel
    cfg['model_type'].to_device(device_list, rank, ddp=ddp)

    # setup optimizer
    if 'optimizer' in cfg:
        params = list(cfg['encoder'].parameters()) + list(cfg['decoder'].parameters())
        cfg['optimizer'] = cfg['optimizer'](params, **cfg['optimizer_kwargs'])
    elif 'g_optimizer' and 'd_optimizer' in cfg:
        g_params = list(cfg['encoder'].parameters()) + list(cfg['decoder'].parameters())
        d_params = list(cfg['discriminator'].parameters())
        cfg['g_optimizer'] = cfg['g_optimizer'](g_params, **cfg['g_optimizer_kwargs'])
        cfg['d_optimizer'] = cfg['d_optimizer'](d_params, **cfg['d_optimizer_kwargs'])

    # setup lr scheduler
    if 'lr_scheduler' in cfg and cfg['lr_scheduler']:
        cfg['lr_scheduler'] = cfg['lr_scheduler'](cfg['optimizer'], cfg['num_epochs'], cfg['warmup_epochs'])
    elif 'g_lr_scheduler' in cfg and cfg['g_lr_scheduler']:
        cfg['g_lr_scheduler'] = cfg['g_lr_scheduler'](cfg['g_optimizer'], cfg['num_epochs'], cfg['warmup_epochs'])
        cfg['d_lr_scheduler'] = cfg['d_lr_scheduler'](cfg['d_optimizer'], cfg['num_epochs'], cfg['warmup_epochs'])
    return cfg['model_type']


def setup_model_darts(cfg, device_list, rank=None, ddp=False):
    cfg['model_type'].to_device(device_list, rank, ddp=ddp)
    model_weights = list(cfg['encoder'].get_weights() + list(cfg['decoder'].parameters()))
    model_alpha = cfg['encoder'].get_alphas()

    # Initialize optimizers
    cfg['w_optimizer'] = cfg['w_optimizer'](model_weights, **cfg['w_optimizer_kwargs'])
    cfg['a_optimizer'] = cfg['a_optimizer'](model_alpha, **cfg['a_optimizer_kwargs'])

    # Initialize lr scheduler
    cfg['w_lr_scheduler'] = cfg['w_lr_scheduler'](cfg['w_optimizer'], cfg['num_epochs'], cfg['warmup_epochs'])
    # cfg['a_lr_scheduler'] = cfg['lr_scheduler'](cfg['a_optimizer'], cfg['num_epochs'], cfg['warmup_epochs'])
    return cfg['model_type']


def log_cfg(cfg, logger, nopause=False):
    logger.write('-------------------------------------------------')
    logger.write('config:')
    template = '\t{0:30}{1}'
    for key in sorted(cfg.keys()):
        logger.write(template.format(key, cfg[key]))
    logger.write('-------------------------------------------------')

    if nopause:
        return
    input('Press Enter to continue...')
    logger.write('-------------------------------------------------')


def print_start_info(cfg, is_training=False):
    model_type = 'training' if is_training else 'testing'
    print("\n--------------- begin {0} ---------------".format(model_type))
    print(f"number of epochs, {cfg['num_epochs']}")
    print(f"batch size, {cfg['batch_size']}\n")
