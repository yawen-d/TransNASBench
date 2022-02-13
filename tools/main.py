# main.py for training a net
import os
import sys
import time
import torch
import random
import argparse
import matplotlib
from pathlib import Path

matplotlib.use("Agg")

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from log_utils import Logger, AverageMeter, ModelDB
from data import TaskonomyDataset, get_datasets

sys.path.remove(str(lib_dir))

import utils


def main(rank, world_size, device_list, args):
    if args.ddp:
        # create default process group
        print(f"rank: {rank}, world_size: {world_size}")
        utils.setup_ddp(rank, world_size)
    print(f'Training network: {args.encoder_str}')

    # setup config
    utils.prepare_seed_cudnn(args.seed)
    cfg = utils.setup_config(args, world_size)

    # pre-run preparation
    log_file_name = f"{cfg['encoder_str']}_batch{cfg['batch_size']}_lr{cfg['initial_lr']}{cfg['plot_msg']}.txt"
    train_logger_path = str((Path(cfg['log_dir']) / log_file_name).resolve())
    checkpoints_dir = str((Path(cfg['log_dir']) / "checkpoints").resolve())
    demo_dir = str((Path(cfg['log_dir']) / "img_output").resolve())
    utils.mkdir_op(cfg['log_dir'], checkpoints_dir, demo_dir)

    # setting up logger & model DB
    logger = Logger(train_logger_path)
    utils.log_cfg(cfg, logger, nopause=args.nopause)

    # build model
    logger.write(f"Using {torch.cuda.device_count()} GPUs!")
    model = utils.setup_model(cfg, device_list, rank, ddp=args.ddp)
    logger.write("--------------- {0:12} setup finished! ---------------".format("Model"))
    model_db = ModelDB(cfg, args.encoder_str, logger, world_size)
    logger.write("--------------- {0:12} setup finished! ---------------".format("ModelDB"))

    try:
        # prepare seed and cudnn
        utils.prepare_seed_cudnn(args.seed, logger)
        logger.write(utils.get_machine_info())

        # data loading
        train_data, val_data, test_data = get_datasets(cfg)

        if args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size,
                                                                            rank=rank, shuffle=False)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg['batch_size'] // world_size,
                                                       shuffle=False, sampler=train_sampler,
                                                       num_workers=cfg['num_workers'] // world_size, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True,
                                                       num_workers=cfg['num_workers'], pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg['batch_size'], shuffle=False,
                                                 num_workers=cfg['num_workers'], pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg['batch_size'], shuffle=False,
                                                  num_workers=cfg['num_workers'], pin_memory=True)

        logger.write("--------------- {0:12} setup finished! ---------------".format("Dataloader"))
        logger.write(f'len(train_data) {len(train_data)}; len(train_loader) {len(train_loader)}')
        logger.write(f'len(val_data) {len(val_data)}; len(val_loader) {len(val_loader)}')
        logger.write(f'len(test_data) {len(test_data)}; len(test_loader) {len(test_loader)}')

        # execute training
        start_train_time = time.time()
        utils.print_start_info(cfg, is_training=True)
        model_db.change_status_save('running')  # ['running', 'finished', 'corrupted']
        model.evaluate_for_seed(cfg, train_loader, val_loader, test_loader, logger, model_db)
        end_train_time = time.time() - start_train_time

        logger.write('time to train %d epochs: %.3f hrs' % (cfg['num_epochs'], end_train_time / (60 * 60)))
        logger.write('avg time per epoch: %.3f hrs' % ((end_train_time / (60 * 60)) / cfg['num_epochs']))
        model_db.change_status_save('finished')

        if args.ddp: utils.cleanup()

    except Exception as e:
        model_db.change_status_save('corrupted')
        logger.write(f"model {args.encoder_str} in {cfg['task_name']} is corrupted!")
        logger.write(e)
        if args.ddp: utils.cleanup()
        raise
    except:
        model_db.change_status_save('corrupted')
        logger.write(f"model {args.encoder_str} in {cfg['task_name']} is corrupted!")
        if args.ddp: utils.cleanup()
        raise


if __name__ == "__main__":
    # Get Arguments
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('cfg_dir', type=str, help='directory containing config.py file')
    parser.add_argument('--encoder_str', type=str, default='resnet50')
    parser.add_argument('--nopause', dest='nopause', action='store_true')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--ddp', action='store_true')
    parser.set_defaults(nopause=True)

    device_list = list(range(torch.cuda.device_count()))
    args = parser.parse_args()
    if args.ddp:  # DDP
        import torch.multiprocessing as mp

        world_size = len(device_list)
        assert world_size > 1
        mp.spawn(main, args=(world_size, device_list, args), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = len(device_list)
        main(rank=rank, world_size=world_size, device_list=device_list, args=args)
