from sklearn.metrics import roc_auc_score
import random
import os
import math
import time
import torch
import numpy as np
import sys
import builtins

def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score

def seed_torch(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def open(file, mode=None, encoding=None):
    if '/' in file:
        dir = os.path.dirname(file)
        if not os.path.isdir(dir): os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f

class Logger(object):
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = open(file, 'w')
    
    def write(self, message, is_terminal = 1, is_file = 1):
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

def create_out_dir(args):
    if not os.path.isdir(f'../{args.model}_{args.image_size}_{args.lr}'):
        os.makedirs(f'../{args.model}_{args.image_size}_{args.lr}')
    return f'../{args.model}_{args.image_size}_{args.lr}'