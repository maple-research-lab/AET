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

import architectures
from dataloader import DataLoader, GenericDataset
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
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')


opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = GenericDataset('imagenet','train',random_sized_crop=False)
dataloader = DataLoader(dataset, batch_size=1, unsupervised=True, shift=28, scale=[0.8,1.2], resample=Image.BILINEAR, fillcolor=(128,128,128))

ngpu = int(opt.ngpu)

opt_net = {'num_classes':8}
net = architectures.create_model(opt_net)
if opt.cuda:
    net = net.cuda()

if opt.net != '':
    net.load_state_dict(torch.load(opt.net)['network'])

print(net)

criterion = nn.MSELoss()
if opt.cuda:
    criterion = criterion.cuda()

net.eval()    
i = 0        
for b in dataloader():
    img1, img2, matrix = b
    if opt.cuda:
        img1 = img1.cuda()
        img2 = img2.cuda()
        matrix = matrix.cuda()
    img1, img2, matrix = torch.autograd.Variable(img1), torch.autograd.Variable(img2), torch.autograd.Variable(matrix)
    
    batch_size = img1.size(0)
    f1, f2, output = net(img1, img2)
    
    err_matrix = criterion(output, matrix)
    err = err_matrix
    
    print('[%d] Loss: %.4f, Loss_matrix: %.4f'
          % (i,
             err.data[0], err_matrix.data[0]))
    
    vutils.save_image(img1.data,
                        './original_image_{}.png'.format(str(i)),
                        normalize=True)
    
    unorm_img = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    to_img = transforms.ToPILImage()
    img1 = unorm_img(img1[0].data)
    img1 = to_img(img1.cpu())
    
    output = output.view(8,1,1).data
    unorm_output = UnNormalize(mean=(0.,0.,112.,0.,0.,112.,6.4e-5,6.4e-5), std=(0.75,0.75,118.,0.75,0.75,118.,7.75e-4, 7.75e-4))
    output = unorm_output(output)
    output = output.view(8).cpu().numpy()
    
    kwargs = {"fillcolor": (0,0,0)} if PILLOW_VERSION[0] == '5' else {}
    img3 = img1.transform((224, 224), Image.PERSPECTIVE, output, PIL.Image.BILINEAR, **kwargs)
    to_tensor = transforms.ToTensor()
    img3 = to_tensor(img3)
    
    matrix = matrix.view(8,1,1).data
    matrix = unorm_output(matrix)
    matrix = matrix.view(8).cpu().numpy()
    
    img2 = img1.transform((224, 224), Image.PERSPECTIVE, matrix, PIL.Image.BILINEAR, **kwargs)
    img2 = to_tensor(img2)
    
    vutils.save_image(img2,
                        './warped_image_{}.png'.format(str(i)),
                        normalize=False)
    
    vutils.save_image(img3,
                        './predicted_image_{}.png'.format(str(i)),
                        normalize=False)
    print(matrix)
    print(output)
    i += 1
    if i >10:
        break
                                                                                    