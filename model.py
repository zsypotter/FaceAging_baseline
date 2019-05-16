import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
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

def VGG16_path4():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def VGG16_path3():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def VGG16_path2():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def VGG16_path1():
    conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    return conv


def Path1():
    model = VGG16_path1()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def Path2():
    model = VGG16_path2()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def Path3():
    model = VGG16_path3()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model


def Path4():
    model = VGG16_path4()
    vgg16 = models.vgg16(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(512),
            nn.Conv2d(512, 1, 4, 2, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1 * 3 * 12, 1)
        )

    def forward(self, path1, path2, path3, path4):
        out1 = self.conv1(path1)
        out2 = self.conv2(path2)
        out3 = self.conv3(path3)
        out4 = self.conv4(path4)
        out = torch.cat((out1, out2, out3, out4), 1)
        out = out.view(-1, 1 * 3 * 12)
        out = self.fc(out)

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
        print("Loading Generator")
        self.G = Generator().to(self.device)
        self.G = nn.DataParallel(self.G, list(range(args.ngpu)))
        print_network(self.G)

        # load VGG
        print("Loading VGG1")
        self.VGG1 = Path1().to(self.device)
        self.VGG1 = nn.DataParallel(self.VGG1, list(range(args.ngpu)))
        print_network(self.VGG1)

        print("Loading VGG2")
        self.VGG2 = Path2().to(self.device)
        self.VGG2 = nn.DataParallel(self.VGG2, list(range(args.ngpu)))
        print_network(self.VGG2)

        print("Loading VGG3")
        self.VGG3 = Path3().to(self.device)
        self.VGG3 = nn.DataParallel(self.VGG3, list(range(args.ngpu)))
        print_network(self.VGG3)

        print("Loading VGG4")
        self.VGG4 = Path4().to(self.device)
        self.VGG4 = nn.DataParallel(self.VGG4, list(range(args.ngpu)))
        print_network(self.VGG4)

        print("Loading Discriminator")
        self.D = Discriminator().to(self.device)
        self.D = nn.DataParallel(self.D, list(range(args.ngpu)))
        print_network(self.D)

        # criterion
        self.mse_criterion = nn.MSELoss()

        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))
        self.G_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, self.lr_decay_step, gamma=self.lr_decay)
        self.D_scheduler = torch.optim.lr_scheduler.StepLR(self.D_optimizer, self.lr_decay_step, gamma=self.lr_decay)

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
            self.D_scheduler.step()
            for iters, data in enumerate(trainloader, 0):
                start_time = time.clock()

                # load data
                young_imgs, old_imgs = data
                young_imgs = young_imgs.to(self.device)
                old_imgs = old_imgs.to(self.device)
                b_size = young_imgs.size(0)
                real_label = torch.ones((b_size, 1)).to(self.device)
                fake_label = torch.zeros((b_size, 1)).to(self.device)

                young_imgs_generator = (young_imgs - generator_mean) / generator_std
                young_imgs_vgg = (young_imgs - vgg_mean) / vgg_std
                old_imgs_vgg = (old_imgs - vgg_mean) / vgg_std
                fake_old_imgs = self.G(young_imgs_generator)
                fake_old_imgs_vgg = ((fake_old_imgs * generator_std + generator_mean) - vgg_mean) / vgg_std

                p1_old_real = self.VGG1(old_imgs_vgg)
                p2_old_real = self.VGG2(old_imgs_vgg)
                p3_old_real = self.VGG3(old_imgs_vgg)
                p4_old_real = self.VGG4(old_imgs_vgg)

                p1_old_fake = self.VGG1(fake_old_imgs_vgg)
                p2_old_fake = self.VGG2(fake_old_imgs_vgg)
                p3_old_fake = self.VGG3(fake_old_imgs_vgg)
                p4_old_fake = self.VGG4(fake_old_imgs_vgg)

                # update G
                self.G.zero_grad()             
                fake_old_imgs_D = self.D(p1_old_fake, p2_old_fake, p3_old_fake, p4_old_fake)
                errG_fake = self.mse_criterion(fake_old_imgs_D, real_label)
                p_loss = self.mse_criterion(young_imgs_generator, fake_old_imgs) / (self.input_size * self.input_size * 3)
                a_loss = errG_fake
                G_loss = self.p_w * p_loss + self.a_w * a_loss
                G_loss.backward()
                self.G_optimizer.step()

                # update D
                self.D.zero_grad()

                # real
                true_old_imgs_D = self.D(p1_old_real.detach(), p2_old_real.detach(), p3_old_real.detach(), p4_old_real.detach())
                errD_real = self.mse_criterion(true_old_imgs_D, real_label)

                # fake
                fake_old_imgs_D = self.D(p1_old_fake.detach(), p2_old_fake.detach(), p3_old_fake.detach(), p4_old_fake.detach())
                errD_fake = self.mse_criterion(fake_old_imgs_D, fake_label)

                D_loss = (errD_real + errD_fake) / 2
                D_loss.backward()
                self.D_optimizer.step()

                end_time = time.clock()
                print('epochs: [{}/{}], iters: [{}/{}], per_iter {:.4f}, G_loss: {:.4f}, D_loss: {:.4f}, lrG: {:.8f}, lrD: {:.8f}'.format(epoch, self.epoch, iters, iters_num, end_time - start_time, errG_fake.item(), D_loss.item(), self.G_optimizer.param_groups[0]['lr'], self.D_optimizer.param_groups[0]['lr']))

                if iters % 10 == 0:
                    writer.add_image("young_imgs_generator", (young_imgs_generator + 1) / 2, iters + iters_num * epoch)
                    writer.add_image("fake_old_imgs", (fake_old_imgs + 1) / 2, iters + iters_num * epoch)
                    writer.add_scalar("P_loss", p_loss, iters + iters_num * epoch)
                    writer.add_scalar("G_loss", errG_fake, iters + iters_num * epoch)
                    writer.add_scalar("D_loss", D_loss, iters + iters_num * epoch)
                

    def test(self):
        print('Testing Success!')