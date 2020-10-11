import os

import numpy as np
import torch


def _get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def get_available_device():
    """
    :return: 'cuda:x' (gpu with more available memory) if available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        return f'cuda:{_get_freer_gpu()}'
    return 'cpu'


def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
