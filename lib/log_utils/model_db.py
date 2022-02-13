import torch
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

from .logger import Logger

lib_dir = (Path(__file__).parent / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from models.model_info import get_inference_time, get_params


class ModelDB(object):
    """ModelDB class for storing experiment results in each run"""
    def __init__(self, cfg, encoder_str, logger, world_size):
        self.cfg = cfg
        self.encoder_str = encoder_str
        self.logger = logger
        self.world_size = world_size

        # Setup paths
        self.encoder_ckpt_path = str(
            Path(cfg['log_dir']) / f'checkpoints/{self.encoder_str}_encoder.pth')
        self.decoder_ckpt_path = str(
            Path(cfg['log_dir']) / f'checkpoints/{self.encoder_str}_decoder.pth')
        if 'discriminator' in self.cfg.keys():
            self.discriminator_ckpt_path = str(
                Path(cfg['log_dir']) / f'checkpoints/{self.encoder_str}_discriminator.pth')

        # Initialize database for current model
        self.model_db_filename = str(Path(cfg['log_dir']) / 'model_db.json')
        self.logger.write(f'model_db_filename {self.model_db_filename}')
        self._initialize_dict()

    def save(self, model_dic, epoch, metrics):
        # Save checkpoints
        for category, model in model_dic.items():
            torch.save(model.state_dict(), str(
                Path(self.cfg['log_dir']) / f'checkpoints/{self.encoder_str}_{category}_{epoch}.pth'))
            torch.save(model.state_dict(), str(
                Path(self.cfg['log_dir']) / f'checkpoints/{self.encoder_str}_{category}_final.pth'))

        # Save metrics
        metric_collection = {}
        assert isinstance(metrics, dict)
        assert len(metrics.keys()) == len(self.cfg['metric_content']), f'Received metrics {metrics.keys()}'
        for key, value in metrics.items():
            if key in self.cfg['metric_content']:
                metric_collection[key] = value

        # Save to database
        with open(self.model_db_filename, 'r') as fp:
            model_db = json.load(fp)
        model_db[self.encoder_str]['metrics'][f'epoch_{epoch}'] = metric_collection
        save_json(model_db, self.model_db_filename)

    def change_status_save(self, status):  # ['running', 'finished', 'corrupted']
        assert status in ['running', 'finished', 'corrupted']
        # Save to database
        with open(self.model_db_filename, 'r') as fp:
            model_db = json.load(fp)
        model_db[self.encoder_str]['finish_train'] = status
        save_json(model_db, self.model_db_filename)

    def print_model_db(self):
        with open(self.model_db_filename, 'r') as fp:
            model_db = json.load(fp)
            print(json.dumps(model_db, indent=4, sort_keys=True))

    def maintain_validity(self):
        """
        The model database is valid when:
        (1) All the model has full information recorded for each epoch in cfg['num_epoch'].
        (2) All the model's checkpoint file exists.
        (3) Each encoder has at least one version of valid training information.
        """
        with open(self.model_db_filename, 'r') as fp:
            model_db = json.load(fp)

        for encoder in list(model_db.keys()):
            for version_str, model_info in list(model_db[encoder].items()):
                valid_version = True
                if len(model_info['metrics'].keys()) != self.cfg['num_epochs']:
                    valid_version = False
                if not valid_version:
                    remove_if_exists(model_info['encoder_ckpt_file'])
                    remove_if_exists(model_info['decoder_ckpt_file'])
                    if 'discriminator_ckpt_file' in model_info.keys():
                        remove_if_exists(model_info['discriminator_ckpt_file'])
                    del model_db[encoder][version_str]
            if len(model_db[encoder].items()) < 1:
                del model_db[encoder]
        save_json(model_db, self.model_db_filename)

    def create_initial_dic(self):
        initial_version_dic = {
            'world_size': self.world_size,
            'finish_train': '',
            'encoder_ckpt_file': self.encoder_ckpt_path,
            'decoder_ckpt_file': self.decoder_ckpt_path,
            'total_params': get_params(self.cfg['encoder']),
            'inference_time': get_inference_time(self.cfg['model_type'], (1, 3, 720, 1080)),
            'metrics': {}
        }
        if 'discriminator' in self.cfg.keys():
            initial_version_dic['discriminator_ckpt_path'] = self.discriminator_ckpt_path
        return initial_version_dic

    def _initialize_dict(self):
        if not os.path.exists(self.model_db_filename):
            initial_dict = {
                self.encoder_str: self.create_initial_dic()
            }
            save_json(initial_dict, self.model_db_filename)
        else:
            # self.maintain_validity()
            initial_version_dic = self.create_initial_dic()
            with open(self.model_db_filename, 'r') as fp:
                model_db = json.load(fp)
            # if self.encoder_str in model_db.keys():
            #     self.logger.write(f'Creating a new version for {self.encoder_str} in database...')
            #     model_db[self.encoder_str][self.version_str] = initial_version_dic
            # else:
            self.logger.write(f'Creating {self.encoder_str} in database...')
            model_db[self.encoder_str] = initial_version_dic
            save_json(model_db, self.model_db_filename)


def save_json(content, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4)


def remove_if_exists(file):
    if os.path.isfile(file):
        os.remove(file)
