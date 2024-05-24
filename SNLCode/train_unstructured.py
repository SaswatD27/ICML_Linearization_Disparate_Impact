import argparse
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures_unstructured import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
from utils import train, test
import sys
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_path = os.path.join(parent_dir, 'src')
sys.path.append(src_path)

from dataset import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str)#, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--lr_milestones', default = [80, 120])
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for weight augmentation")
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
parser.add_argument('--seed', type=int, default=1, help='data split seed')
parser.add_argument('--imbalance', type=int, default=0)
parser.add_argument('--ngroups', type=int, default=5)
parser.add_argument('--minorityprop', type=float, default=0.1)

args = parser.parse_args()


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)
    is_celeba = False
    if args.dataset.lower() == 'cifar10':
        args.ngroups = 10
        if args.imbalance == 1:
            class_perc_dict = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: args.minorityprop, 6: args.minorityprop, 7: args.minorityprop, 8: args.minorityprop, 9: args.minorityprop}
        else:
            class_perc_dict = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
    if args.dataset.lower() == 'svhn':
        args.ngroups = 10
        if args.imbalance == 1:
            class_perc_dict = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: args.minorityprop, 6: args.minorityprop, 7: args.minorityprop, 8: args.minorityprop, 9: args.minorityprop}
        else:
            class_perc_dict = {i:1 for i in range(10)}#{0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.3, 9: 0.3}
    if args.dataset.lower() == 'cifar100':
        args.ngroups = 100
        if args.imbalance == 1:
            class_perc_dict = {i: 0.9 for i in range(50)}
            class_perc_dict.update({i: args.minorityprop for i in range(50, 100)})
        else:
            class_perc_dict = {i: 0.9 for i in range(50)}
            class_perc_dict.update({i: 0.3 for i in range(50, 100)})
    elif args.dataset.lower() == 'utkface_age':
        args.ngroups = 5
        if args.imbalance == 1:
            class_perc_dict = {0: args.minorityprop, 1: 1, 2: args.minorityprop, 3: 1, 4: 1}
        else:
            class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    elif args.dataset.lower() == 'utkface_race':
        args.ngroups = 5
        if args.imbalance == 1:
            class_perc_dict = {0: 1, 1: 1, 2: args.minorityprop, 3: 1, 4: args.minorityprop}
        else:
            class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    elif args.dataset.lower() == 'celeba':
        is_celeba = True
        args.ngroups = 4
        class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1}#, 4: 1 , 5: 1, 6: 1, 7: 1}
    elif args.dataset.lower() == 'celeba_unbalanced':
        is_celeba = True
        args.ngroups = 2
        class_perc_dict = {0: 0.9, 1: 0.1}
    
    train_dataset, test_dataset, leftout_training_set, leftout_test_set = unbalance_dataset(
args.dataset, class_perc_dict=class_perc_dict, seed=args.seed, data_dir = "/ReLU_Reduction_Fairness-main/DeepReduce/data", is_celeba = is_celeba)
    
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset, device, args)

    for name, param in model.named_parameters():
        if 'alpha' in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[args.epochs // 2,  3*args.epochs // 4], last_epoch=-1)

    best_top1 = 0.

    for epoch in range(args.epochs):

        # training
        train(train_loader, model, criterion, optimizer, epoch, device)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        _, top1, _ = test(test_loader, model, criterion, device, cur_step)
        scheduler.step()

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False

        if is_best:
            torch.save({
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'best_checkpoint.pth.tar'))

        print("")

    print("Best model's validation acc: {:.4%}".format(best_top1 / 100))
if __name__ == "__main__":
    main()

