import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import os
import numpy
import time
from tensorboardX import SummaryWriter
from data_loader import customData
from utils import *

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        out = residual + x
        out = self.relu(out)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.in_channels = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 9, 1, 4),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 512, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(128, 512, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(128, 3, 9, 1, 4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)

        return out

class Aging_Model(object):
    def __init__(self, args, model_name):

        # H_parameter
        self.dataroot = args.dataroot
        self.dataset = args.dataset
        self.age_part = args.age_part
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.ngpu = args.ngpu
        self.input_size = args.input_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.p_w = args.p_w
        self.a_w = args.a_w
        self.i_w = args.i_w
        self.lrG = args.lrG
        self.lrD = args.lrD
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.random_seed = args.random_seed
        self.model_name = model_name
        self.lr_decay_step = args.lr_decay_step
        self.lr_decay = args.lr_decay

        # set gpu
         # set gpu device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

        # network
        self.G = Generator().to(self.device)
        self.G = nn.DataParallel(self.G, list(range(args.ngpu)))
        print_network(self.G)

        # criterion
        self.mse_criterion = nn.MSELoss(reduce=False)

        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.G_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, self.lr_decay_step, gamma=self.lr_decay)

    def train(self):
        # set random seed
        print(self.model_name)
        setup_seed(self.random_seed)
        print("Set random seed", self.random_seed)

        # prepare data
        young_path = os.path.join(self.dataroot, self.dataset, '0')
        old_path = os.path.join(self.dataroot, self.dataset, self.age_part)
        print(young_path, old_path)

        data_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            ])

        trainset = customData(young_path, old_path, data_transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        generator_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        generator_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        # prepare log
        log_path = os.path.join(self.log_dir, self.model_name)
        writer = SummaryWriter(log_path)

        # train loop
        print("Starting Training Loop...")
        iters_num = len(trainloader)
        self.G.train()
        for epoch in range(self.epoch):
            self.G_scheduler.step()
            for iters, data in enumerate(trainloader, 0):
                start_time = time.clock()

                # load data
                young_imgs, old_imgs = data
                young_imgs = young_imgs.to(self.device)
                old_imgs = old_imgs.to(self.device)
                young_imgs_generator = (young_imgs - generator_mean) / generator_std
                young_imgs_vgg = (young_imgs - vgg_mean) / vgg_std

                # update G
                self.G.zero_grad()
                fake_old_imgs = self.G(young_imgs_generator)
                G_loss = self.mse_criterion(young_imgs_generator, fake_old_imgs) / (self.input_size * self.input_size * 3)
                G_loss.backward()
                self.G_optimizer.step()

                end_time = time.clock()
                print('epochs: [{}/{}], iters: [{}/{}], per_iter {:.4f}, G_loss: {:.4f}, lrG: {:.8f}'.format(epoch, self.epoch, iters, iters_num, end_time - start_time, G_loss.item(), self.G_optimizer.param_groups[0]['lr']))

                if iters % 10 == 0:
                    writer.add_image("young_imgs_generator", (young_imgs_generator + 1) / 2, iters + iters_num * epoch)
                    writer.add_image("fake_old_imgs", (fake_old_imgs + 1) / 2, iters + iters_num * epoch)
                    writer.add_scalar("G_loss", G_loss, iters + iters_num * epoch)
                

    def test(self):
        print('Testing Success!')