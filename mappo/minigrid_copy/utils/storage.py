import logging
import os
import sys
import collections
import numpy as np
import torch

def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
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