import os
import random
import numpy as np
import torch


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize(embeddings, dim=1, **kwargs):
    return torch.nn.functional.normalize(embeddings, p=2, dim=dim, **kwargs)
