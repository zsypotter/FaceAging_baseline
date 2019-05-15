import argparse
import os
import torch
import time
from model import Aging_Model

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='/data2/zhousiyu/dataset/')
parser.add_argument("--dataset", type=str, default='CACD2000', choices=['CACD2000', 'MORPH'])
parser.add_argument("--age_part", type=str, default='3', choices=['1', '2', '3'])
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument('--save_dir', type=str, default='Pyramid-GAN/Dict/', help='Directory name to save the model')
parser.add_argument('--result_dir', type=str, default='Pyramid-GAN/results/', help='Directory name to save the generated images')
parser.add_argument('--log_dir', type=str, default='Pyramid-GAN/runs/')
parser.add_argument('--p_w', type=float, default=0.2)
parser.add_argument('--a_w', type=float, default=750)
parser.add_argument('--i_w', type=float, default=0.0005)
parser.add_argument('--lrG', type=float, default=0.0001)
parser.add_argument('--lrD', type=float, default=0.0001)
parser.add_argument('--lr_decay_step', type=int, default=1)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--random_seed', type=int, default=999)
args = parser.parse_args()

model_name = args.dataset + '_' + str(args.batch_size) + '_' + str(args.p_w) + '_' + str(args.a_w) + '_' + str(args.i_w) + '_' + time.asctime(time.localtime(time.time()))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

aging_model = Aging_Model(args, model_name)

aging_model.train()
aging_model.test()