'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import h5py

from NetworkInNetwork import Regressor
from NonLinearClassifier import Classifier

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswth("__")
#    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--net', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')                    
# Architecture
#parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
#                    choices=model_names,
#                    help='model architecture: ' +
#                        ' | '.join(model_names) +
#                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#parser.add_argument('--Ddim', type=int, default=4)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    

best_acc = 0  # best test accuracy

def main():
    global best_acc
    global device
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='/home/lhzhang/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

    testset = dataloader(root='/home/lhzhang/cifar10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    net = Regressor(_num_stages=4, _use_avg_on_conv3=False).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    if args.net != '':
        net.load_state_dict(torch.load(args.net))

#    model = Classifier(_nChannels=192*8*8, _num_classes=10, _cls_type='MultLayer').to(device)
#    model = torch.nn.DataParallel(model, device_ids=[0])
#    model = Classifier(_nChannels=96*16*16, _num_classes=10, _cls_type='MultLayer').to(device)

    cudnn.benchmark = True
#    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

#    criterion = nn.CrossEntropyLoss()
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    # Resume
#    title = 'cifar-10-'
#    if args.resume:
#        # Load checkpoint.
#        print('==> Resuming from checkpoint..')
#        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
#        args.checkpoint = os.path.dirname(args.resume)
#        checkpoint = torch.load(args.resume)
#        best_acc = checkpoint['best_acc']
#        start_epoch = checkpoint['epoch']
#        model.load_state_dict(checkpoint['state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer'])
#        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
#    else:
#        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
#        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


#    if args.evaluate:
#        print('\nEvaluation only')
#        test_loss, test_acc = test(testloader, net, model, criterion, start_epoch, use_cuda, device)
#        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
#        return

    # Train and val
    net.eval()
    train_feats = []
    train_labels = []
    for i, data in enumerate(testloader, 0):
        inputs = data[0].to(device)
        targets = data[1].to(device)
        batch_size = inputs.size(0)
        f1, f2 = net(inputs, inputs, out_feat_keys=['conv2'])
        f1 = nn.functional.avg_pool2d(f1, 2)
        f1 = f1.view(-1).cpu().detach().numpy()
        targets = targets.view(-1).cpu().detach().numpy()
        train_feats += [f1]
        train_labels += [targets]
        print(i)
        
    train_feats = np.asarray(train_feats, dtype='float32')
    train_labels = np.asarray(train_labels, dtype='float32')
    
    print(train_feats.shape)
    print(train_labels.shape)
    with h5py.File('./features/test_pool2.h5', 'w') as hf:
        hf.create_dataset('feats', data=train_feats)
        hf.create_dataset('labels', data=train_labels)    
    
    
    
#    for epoch in range(start_epoch, args.epochs):
#        adjust_learning_rate(optimizer, epoch)
#
#        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
#
#        train_loss, train_acc = train(trainloader, net, model, criterion, optimizer, epoch, use_cuda, device)
#        test_loss, test_acc = test(testloader, net, model, criterion, epoch, use_cuda, device)
#
#        # append logger file
#        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
#
#        # save model
#        is_best = test_acc > best_acc
#        best_acc = max(test_acc, best_acc)
#        save_checkpoint({
#                'epoch': epoch + 1,
#                'state_dict': model.state_dict(),
#                'acc': test_acc,
#                'best_acc': best_acc,
#                'optimizer' : optimizer.state_dict(),
#            }, is_best, checkpoint=args.checkpoint)
#
#    logger.close()
#    logger.plot()
#    savefig(os.path.join(args.checkpoint, 'log.eps'))
#
#    print('Best acc:')
#    print(best_acc)

def train(trainloader, net, model, criterion, optimizer, epoch, use_cuda, device):
    # switch to train mode
    net.train()
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
            
#        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        f1, f2 = net(inputs, inputs, out_feat_keys=['conv2'])
        outputs = model(f1)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, net, model, criterion, epoch, use_cuda, device):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
            
#        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
#        eye = torch.autograd.Variable(eye)

        # compute output
        f1, f2 = net(inputs, inputs, out_feat_keys=['conv2'])
        outputs = model(f1)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
