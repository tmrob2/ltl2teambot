import logging
import os
import sys
import collections
import numpy as np
import torch
import math

def synthesize(array, multi_obj=False, n_agents=1):
    d = collections.OrderedDict()
    if not multi_obj:
        d["mean"] = np.mean(array)
        d["std"] = np.std(array)
        d["min"] = np.amin(array)
        d["max"] = np.amax(array)
    else:
        mean = []
        for i in range(n_agents):
            x = np.mean(array[i], 0).tolist()
            mean.append(list(map(lambda n: truncate(n, 3), x)))
        d["mean"] = mean
        #d["mean"] = [np.mean(array[i], 0).tolist() for i in range(n_agents)]
        #d["std"] = np.std(array, axis=1)
        #d["min"] = np.amin(array, axis=1)
        #d["max"] = np.amax(array, axis=1)
    return d

def get_txt_logger():
    #path = os.path.join(model_dir, "log.txt")
    #utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            #logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor