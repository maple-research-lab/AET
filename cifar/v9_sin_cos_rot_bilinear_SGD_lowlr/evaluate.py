from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from NetworkInNetwork import Regressor
from dataset import CIFAR10


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/home/liheng/cifar10', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--optimizer', default='', help="path to optimizer (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--rot', type=float, default=180)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = CIFAR10(root=opt.dataroot, degrees=opt.rot, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15, fillcolor=(128,128,128), download=True,
                           transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))
                           
test_dataset = CIFAR10(root=opt.dataroot, degrees=opt.rot, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15, fillcolor=(128,128,128), download=True, train=False,
                           transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))


assert train_dataset
assert test_dataset
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)

net = Regressor(_num_stages=4, _use_avg_on_conv3=False).to(device)

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

print(net)

criterion = nn.MSELoss()

# setup optimizer
optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(opt.optimizer))

net.eval()
for epoch in range(opt.niter):
    for i, data in enumerate(train_dataloader, 0):
#        net.zero_grad()
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        matrix = data[2].to(device)
        batch_size = img1.size(0)
        f1, f2, output = net(img1, img2)
        
        err_matrix = criterion(output, matrix)
        err = err_matrix
        
        err_entry = torch.sum((output - matrix)**2, dim=0) / batch_size
#        err.backward()
#        optimizer.step()
        
        print('[%d/%d][%d/%d] Loss: %.4f, Loss_matrix: %.4f'
              % (epoch, opt.niter, i, len(train_dataloader),
                 err.item(), err_matrix.item()))
        print(err_entry)
        print(matrix)
        print(output)
                                                                                    

    # do checkpointing
#    if epoch % 100 == 99:
#        torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch))
#        torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch))