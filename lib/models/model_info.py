import os
import time
import torch
import numpy as np


def get_flops(model, input_size):
    from thop import profile
    input = torch.randn(input_size).to('cuda:0')
    macs, params = profile(model, inputs=(input,))
    return macs, params

def get_params(model):
    return sum(p.numel() for p in model.parameters())

def get_inference_time(model, input_size):
    """
    :param model: need to be on cuda
    :param input_size: input image size (1, 3, 720, 1080)
    :return: inference_time
    """
    # cuDnn configurations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random_input = torch.randn(input_size).to('cuda:0')

    try:
        model.train(False)
    except:
        pass

    time_list = []
    for i in range(1000):
        torch.cuda.synchronize()
        tic = time.time()
        model.forward(random_input)
        torch.cuda.synchronize()
        time_list.append(time.time()-tic)
    time_list = time_list[100:-100]
    inference_time = np.mean(time_list)
    print("Average inference time: {}s".format(inference_time))
    return inference_time
