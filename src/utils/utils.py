import json
from typing import Callable, List

from gym import Env
import numpy as np
import torch
from tqdm import tqdm


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar array x.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    mean = np.mean(x)
    std = np.std(x)
    if with_min_and_max:
        max_global = np.max(x) if len(x) else np.inf
        min_global = np.min(x) if len(x) else np.inf
        return mean, std, min_global, max_global
    else:
        return mean, std


def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
