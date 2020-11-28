"""This file contains helper functions for logging and saving, loading 
tensors seamlessly. This file was added by me."""

import os
import os.path
import torch

def add_to_log(msg, file):
    with open(os.path.join(os.environ['HOME'], file + ".txt"), 'a') as f:
        f.write(msg + "\n")

def pickle_tensor(tensor, file):
    torch.save(tensor, file)

def load_tensor(file):
    return torch.load(file)