# taskonomy_dataset.py defines the TaskonomyDataset class
import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms, utils

from . import load_ops

DOMAIN_DATA_SOURCE = {
    'rgb': ('rgb', 'png'),
    'autoencoder': ('rgb', 'png'),
    'class_object': ('class_object', 'npy'),
    'class_scene': ('class_scene', 'npy'),
    'normal': ('normal', 'png'),
    'room_layout': ('room_layout', 'npy'),
    'segmentsemantic': ('segmentsemantic', 'png'),
    'jigsaw': ('rgb', 'png'),
}


class TaskonomyDataset(Dataset):
    def __init__(self, json_path, dataset_dir, domain, target_load_fn, target_load_kwargs=None, transform=None):
        """
        Loading Taskonomy Datasets.
        Args:
            json_path: /path/to/json_file for train/val/test_filenames (specify which buildings to include)
            dataset_dir: /path/to/dataset/..
            domain: domain
            img_transform_fn : img_transform_fn
            label_transform_fn : label_transform_fn
        """
        self.dataset_dir = dataset_dir
        self.domain = domain
        self.label_type = DOMAIN_DATA_SOURCE[self.domain][1]
        self.all_templates = get_all_templates(dataset_dir, json_path)
        self.target_load_fn = target_load_fn
        self.target_load_kwargs = target_load_kwargs
        self.transform = transform

    def __len__(self):
        return len(self.all_templates)

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()
            template = os.path.join(self.dataset_dir, self.all_templates[idx])
            image = io.imread('.'.join([template.format(domain='rgb'), 'png']))
            label = self.get_label(template)
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)
        except:
            template = os.path.join(self.dataset_dir, self.all_templates[idx])
            raise Exception(f"Error for img {'.'.join([template.format(domain='rgb'), 'png'])}")
        return sample

    def get_label(self, template):
        label_path = '.'.join([template.format(domain=DOMAIN_DATA_SOURCE[self.domain][0]),
                               DOMAIN_DATA_SOURCE[self.domain][1]])
        label = self.target_load_fn(label_path, **self.target_load_kwargs)
        return label


def get_all_templates(dataset_dir, filenames_path):
    """
    Get all templates.
    :param dataset_dir: the dir containing the taskonomy dataset
    :param filenames_path: /path/to/json_file for train/val/test_filenames (specifies which buildings to include)
    :return: a list of absolute paths of all templates
        e.g. "{building}/{domain}/point_0_view_0_domain_{domain}"
    """
    building_lists = load_ops.read_json(filenames_path)['filename_list']
    all_template_paths = []
    for building in building_lists:
        all_template_paths += load_ops.read_json(os.path.join(dataset_dir, building))
    for i, path in enumerate(all_template_paths):
        f_split = path.split('.')
        if f_split[-1] in ['npy', 'png']:
            all_template_paths[i] = '.'.join(f_split[:-1])
    return all_template_paths


def get_datasets(cfg):
    """Getting the train/val/test dataset"""
    train_data = TaskonomyDataset(os.path.join(cfg['data_split_dir'], cfg['train_filenames']),
                                  cfg['dataset_dir'], cfg['task_name'], cfg['target_load_fn'],
                                  target_load_kwargs=cfg['target_load_kwargs'],
                                  transform=cfg['train_transform_fn'])
    val_data = TaskonomyDataset(os.path.join(cfg['data_split_dir'], cfg['val_filenames']),
                                cfg['dataset_dir'], cfg['task_name'], cfg['target_load_fn'],
                                target_load_kwargs=cfg['target_load_kwargs'],
                                transform=cfg['val_transform_fn'])
    test_data = TaskonomyDataset(os.path.join(cfg['data_split_dir'], cfg['test_filenames']),
                                 cfg['dataset_dir'], cfg['task_name'], cfg['target_load_fn'],
                                 target_load_kwargs=cfg['target_load_kwargs'],
                                 transform=cfg['val_transform_fn'])
    return train_data, val_data, test_data
