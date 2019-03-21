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
import PIL
from PIL import Image, PILLOW_VERSION

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/home/lhzhang/cifar10', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--manualSeed', type=int, default=0,help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
                           transform_pre=transforms.Compose([
#                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]))
                           
test_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
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
if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=range(ngpu))

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

print(net)

criterion = nn.MSELoss()


net.eval()            
for i, data in enumerate(train_dataloader, 0):
    img1 = data[0].to(device)
    img2 = data[1].to(device)
    matrix = data[2].to(device)
    matrix = matrix.view(-1, 8)
    
    batch_size = img1.size(0)
    f1, f2, output = net(img1, img2)
    
    err_matrix = criterion(output, matrix)
    err = err_matrix
    
    print('[%d/%d] Loss: %.4f, Loss_matrix: %.4f'
          % (i, len(train_dataloader),
             err.item(), err_matrix.item()))
    
    vutils.save_image(img1,
                        './original_image_{}.png'.format(str(i)),
                        normalize=True)
    
    unorm_img = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    to_img = transforms.ToPILImage()
    img1 = unorm_img(img1[0])
    img1 = to_img(img1.cpu())
    
    output = output.view(8,1,1)
    unorm_output = UnNormalize(mean=(0., 0., 16., 0., 0., 16., 0., 0.), std=(1., 1., 20., 1., 1., 20., 0.015, 0.015))
    output = unorm_output(output)
    output = output.view(8).cpu().detach().numpy()
    
    kwargs = {"fillcolor": (0,0,0)} if PILLOW_VERSION[0] == '5' else {}
    img3 = img1.transform((32, 32), Image.PERSPECTIVE, output, PIL.Image.BILINEAR, **kwargs)
    to_tensor = transforms.ToTensor()
    img3 = to_tensor(img3)
    
    matrix = matrix.view(8,1,1)
    matrix = unorm_output(matrix)
    matrix = matrix.view(8).cpu().detach().numpy()
    
    img2 = img1.transform((32, 32), Image.PERSPECTIVE, matrix, PIL.Image.BILINEAR, **kwargs)
    img2 = to_tensor(img2)
    
    vutils.save_image(img2,
                        './warped_image_{}.png'.format(str(i)),
                        normalize=False)
    
    vutils.save_image(img3,
                        './predicted_image_{}.png'.format(str(i)),
                        normalize=False)
    print(matrix)
    print(output)
    if i >20:
        break
                                                                                    