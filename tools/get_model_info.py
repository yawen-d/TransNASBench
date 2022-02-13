# get_model_info.py for getting model inference time and basic info
import os
import sys
import time
import json
import torch
import random
import argparse
import pandas as pd
from pathlib import Path

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from models.model_info import get_inference_time, get_params, get_flops
sys.path.remove(str(lib_dir))

import utils


parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=8000)
parser.add_argument('--cfg_dir', type=str, default='')
parser.add_argument('--encoder_str', type=str, default='')
parser.add_argument('--nopause', dest='nopause', action='store_true')
parser.add_argument('--seed', type=int, default=666)
parser.set_defaults(nopause=True)


def main(task, rank, world_size, device_list, args):
    # setup config
    utils.prepare_seed_cudnn(args.seed)
    cfg = utils.setup_config(args, world_size)

    # build model
    model = utils.setup_model(cfg, device_list, rank)

    latency_input = (1, 3, 720, 1080)
    inference_time = get_inference_time(model, latency_input)
    # inference_time = ""
    flops_input = (1, 3, 224, 224) if task != "jigsaw" else (1, 9, 3, 64, 64)
    model_params = "" # get_params(model)
    encoder_params = "" # get_params(model.encoder)
    model_FLOPs, _ = "", "" # get_flops(model, flops_input)
    if task != "jigsaw":
        encoder_FLOPs, _ = "", "" # get_flops(model.encoder, flops_input)
    else:
        encoder_FLOPs, _ = "", "" # get_flops(model.encoder, (1, 3, 64, 64))

    return {
        "inference_time": inference_time,
        "model_params": model_params,
        "encoder_params": encoder_params,
        "model_FLOPs": model_FLOPs,
        "encoder_FLOPs": encoder_FLOPs
    }


def save_json(content, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4)


def read_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


if __name__ == '__main__':
    device_list = list(range(torch.cuda.device_count()))
    args = parser.parse_args()

    tasks = ["class_object", "jigsaw", "class_scene", "room_layout", "autoencoder", "normal", "segmentsemantic"]

    root_dir = str((Path(__file__).parent / '..' / '..' / '..').resolve())

    encoder_str_path = os.path.join(root_dir, "transnas-bench/configs/benchmark_cfg/net_strings.json")
    encoder_strs_ss = read_json(encoder_str_path)
    encoder_strs = encoder_strs_ss['macro'] + encoder_strs_ss['micro']

    start, end = args.start, args.end

    all_dict = {}
    save_name = os.path.join(root_dir, f"transnas-bench/benchmark_results/models_latency_{start}-{end}.json")
    for id, encoder in enumerate(encoder_strs):
        print(f"===== [{id}/{start}-{end}, {encoder}] =============")
        if not (start <= id < end):
            continue

        all_dict[encoder] = {}
        for task in tasks:
            args.cfg_dir = os.path.join(root_dir, "transnas-bench/configs/task_cfg/train_from_scratch/", task)
            args.encoder_str = encoder

            rank = 0
            world_size = len(device_list)
            all_dict[encoder][task] = main(task, rank=rank, world_size=world_size, device_list=device_list, args=args)
            print(task, all_dict[encoder][task])
        save_json(all_dict, save_name)

    print("all_dict", len(all_dict.keys()))
    save_json(all_dict, save_name)

