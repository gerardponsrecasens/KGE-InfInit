import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import random
from prettytable import PrettyTable
from torch.nn.init import xavier_normal_
from torch.nn import Parameter
import numpy as np
from copy import deepcopy
import sys, os
from torch.backends import cudnn


def get_param(shape):
    '''create learnable parameters'''
    param = Parameter(torch.Tensor(*shape)).double()
    xavier_normal_(param.data)
    return param


def same_seeds(seed):
    '''Set seed for reproduction'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_fact(path):
    '''
    Load (sub, rel, obj) from file 'path'.
    :param path: xxx.txt
    :return: fact list: [(s, r, o)]
    '''
    facts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            s, r, o = line[0], line[1], line[2]
            facts.append((s, r, o))
    return facts


def build_edge_index(s, o):
    '''build edge_index using subject and object entity'''
    index = [s + o, o + s]
    return torch.LongTensor(index)

""" Calculate infoNCE loss """
""" nodes: number for positive """
def infoNCE(embeds1, embeds2, nodes, temp=0.1):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
	return (-torch.log(nume / deno)).mean()