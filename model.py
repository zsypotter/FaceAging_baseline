import torch
import torch.nn as nn
import os
import numpy
import time
from tensorboardX import SummaryWriter

class Aging_Model(object):
    def __init__(self, args, model_name):
        self.dataroot = args.dataroot
        self.dataset = args.dataset
        self.age_part = args.age_part
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.p_w = args.p_w
        self.a_w = args.a_w
        self.i_w = args.i_w
        self.lrG = args.lrG
        self.lrD = args.lrD
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.random_seed = args.random_seed
        self.model_name = model_name

    def train(self):
        young_files = os.path.join(self.dataroot, self.dataset, '0')
        old_files = os.path.join(self.dataroot, self.dataset, self.age_part)
        print(young_files, old_files)
        print(self.model_name)

    def test(self):
        print('Testing Success!')