# Selective Network Linearization unstructured method.
# Starting from the pretrained model. 

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from architectures_unstructured import ARCHITECTURES, get_architecture
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
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
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--finetune_epochs', default=100, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--alpha', default=1e-5, type=float,
                    help='Lasso coefficient')
parser.add_argument('--threshold', default=1e-2, type=float)
parser.add_argument('--budget_type', default='absolute', type=str, choices=['absolute', 'relative'])
parser.add_argument('--relu_budget', default=50000, type=float)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--stride', type=int, default=1, help='conv1 stride')
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument('--imbalance', type=int, default=0)
parser.add_argument('--minorityprop', type=float, default=0.1)
parser.add_argument('--mu', type=float, default=0.001)
parser.add_argument('--ngroups', type=int, default=5)
args = parser.parse_args()

ngroups = args.ngroups

if args.budget_type == 'relative' and args.relu_budget > 1:
    print(f'Warning: relative budget type is used, but the relu budget is {args.relu_budget} > 1.')
    sys.exit(1)

def relu_counting(net, args):
    relu_count = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            relu_count += (boolean_list == 1).sum()
    return relu_count

def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)

    logfilename = os.path.join(args.outdir, args.logname)

    log(logfilename, "Hyperparameter List")
    log(logfilename, "Finetune Epochs: {:}".format(args.finetune_epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "ReLU Budget: {:}".format(args.relu_budget))
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
            class_perc_dict = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.3, 9: 0.3}
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
        class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1}#, 4: 1}#, 5: 1, 6: 1, 7: 1}
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


    # Loading the base_classifier
    base_classifier = get_architecture(args.arch, args.dataset, device, args)
    checkpoint = torch.load(args.savedir, map_location=device)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()

    log(logfilename, "Loaded the base_classifier")

    # Calculating the loaded model's test accuracy.
    original_acc = model_inference(base_classifier, test_loader,
                                    device, display=True)
    
    log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))

    # Creating a fresh copy of network not affecting the original network.
    net = copy.deepcopy(base_classifier)
    net = net.to(device)

    relu_count = relu_counting(net, args)

    log(logfilename, "Original ReLU Count: {}".format(relu_count))

    # Alpha is the masking parameters initialized to 1. Enabling the grad.
    for name, param in net.named_parameters():
        if 'alpha' in name:
            param.requires_grad = True
        
    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = Adam(net.parameters(), lr=args.lr)
    
    # counting number of ReLU.
    total = relu_counting(net, args)
    if args.budget_type == 'relative':
        args.relu_budget = int(total * args.relu_budget)

    # Corresponds to Line 4-9
    lowest_relu_count, relu_count = total, total
    for epoch in range(args.epochs):
        
        # Simultaneous tarining of w and alpha with KD loss.
        train_loss = mask_train_kd_unstructured(train_loader, net, base_classifier, criterion, optimizer,
                                epoch, device, alpha=args.alpha, display=False)
        acc = model_inference(net, test_loader, device, display=False)

        # counting ReLU in the neural network by using threshold.
        relu_count = relu_counting(net, args)        
        log(logfilename, 'Epochs: {}\t'
              'Test Acc: {}\t'
              'Relu Count: {}\t'
              'Alpha: {:.6f}\t'.format(
                  epoch, acc, relu_count, args.alpha
              )
              )
        
        if relu_count < lowest_relu_count:
            lowest_relu_count = relu_count 
        
        elif relu_count >= lowest_relu_count and epoch >= 5:
            args.alpha *= 1.1

        if relu_count <= args.relu_budget:
            print("Current epochs breaking loop at {:}".format(epoch))
            break

    log(logfilename, "After SNL Algorithm, the current ReLU Count: {}, rel. count:{}".format(relu_count, relu_count/total))

    # Line 11: Threshold and freeze alpha
    for name, param in net.named_parameters():
        if 'alpha' in name:
            boolean_list = param.data > args.threshold
            param.data = boolean_list.float()
            param.requires_grad = False

 
    # Line 12: Finetuing the network
    finetune_epoch = args.finetune_epochs

    optimizer = SGD(net.parameters(), lr=1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction = 'none').to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)
    
    print("Finetuning the model")
    log(logfilename, "Finetuning the model")

    best_top1 = 0
    groups = list(range(ngroups))
    gamma = [0]*ngroups
    mu = args.mu
    for epoch in range(finetune_epoch):
        train_loss, train_top1, train_top5, viol = train_kd_mitigate(train_loader, net, base_classifier, optimizer, criterion, epoch, device, gamma, groups)
        test_loss, test_top1, test_top5 = test_mitigate(test_loader, net, criterion, groups, device, 100, display=True)
        scheduler.step()

        gamma = [gamma[g] + mu*viol[g] for g in groups]
        
        if best_top1 < test_top1:
            best_top1 = test_top1
            is_best = True
        else:
            is_best = False

        if is_best:
            torch.save({
                    'arch': args.arch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, f'snl_best_checkpoint_{args.arch}_{args.dataset}_{args.relu_budget}.pth.tar'))

    print("Final best Prec@1 = {}%".format(best_top1))
    log(logfilename, "Final best Prec@1 = {}%".format(best_top1))
        
if __name__ == "__main__":
    main()
