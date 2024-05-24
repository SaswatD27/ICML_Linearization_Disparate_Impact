from argparse import ArgumentParser
import argparse
from models.utils import load_model
from train_DeepReduce import train_DeepReduce
#from train_DeepReduce_debug import train_DeepReduce
from train_Kundu import train_Kundu
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import logging
import os
import sys
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_path = os.path.join(parent_dir, 'src')
sys.path.append(src_path)

from dataset import *

def reshape_and_create_loader(original_loader):
    # Lists to store reshaped data and labels
    reshaped_data_list = []
    labels_list = []

    # Iterate over the original data loader to get one batch at a time
    for data, labels in original_loader:
        # Reshape the data to have the correct shape
        reshaped_data = data.permute(0, 3, 1, 2)
        
        # Append reshaped data and labels to lists
        reshaped_data_list.append(reshaped_data)
        labels_list.append(labels)

    # Concatenate the reshaped data and labels
    reshaped_data_tensor = torch.cat(reshaped_data_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    # Create a new TensorDataset with reshaped data and labels
    reshaped_dataset = TensorDataset(reshaped_data_tensor, labels_tensor)

    # Create a new DataLoader without explicitly specifying shuffle
    reshaped_loader = DataLoader(reshaped_dataset, batch_size=original_loader.batch_size)

    return reshaped_loader

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--tnet', type=str, default='resnet18_deepreduce_teacher_cifar10')
    args.add_argument('--net', type=str, default='resnet18_deepreduce_student_cifar10')
    args.add_argument('--dataset', type=str, default='cifar10') #UTKFace_age
    args.add_argument('--culled_mask', type=int, default=1)# or 14 or 12 or 123 or 234 or 23 or 124 or 34 #[False, True, True, True])
    #args.add_argument('--thinned', type=bool, default=False)
    args.add_argument('--thinned', type=int, default=0) #0 for False, 1 for True
    args.add_argument('--alpha', type=float, default=1)
    args.add_argument('--rho', type=float, default=1)
    args.add_argument('--kd_mode', type=str, default="st")
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--lr', type=float, default=0.001)
    args.add_argument('--n_epochs', type=int, default=200)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=5e-4)
    args.add_argument('--lambda_kd', type=float, default=0.5)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch', type=int, default=512)
    args.add_argument('--savepath', type=str, default='./saved_models/deepreduce_checkpoint.pth')
    args.add_argument('--mu', type=float, default=0.001)
    args.add_argument('--imbalance', type=int, default=0)
    args.add_argument('--minorityprop', type=float, default=0.1)

    args = args.parse_args()

    #from torchinfo import summary as summary_ #debug
    #summary_list = summary_(smodel, input_size=(1,3,32,32)).summary_list #debug
    #print(summary_list) #debug
    is_celeba = False
    if args.dataset.lower() == 'cifar10':
        ngroups = 10
        class_perc_dict = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
    if args.dataset.lower() == 'cifar100':
        ngroups = 100
        class_perc_dict = {i: 0.9 for i in range(50)}
        class_perc_dict.update({i: 0.3 for i in range(50, 100)})
    if args.dataset.lower() == 'cifar10_balanced':
        ngroups = 10
        class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
        args.dataset = 'cifar10'
    elif args.dataset.lower() == 'utkface_age':
        ngroups = 5
        if args.imbalance == 1:
            class_perc_dict = {0: args.minorityprop, 1: 1, 2: args.minorityprop, 3: 1, 4: 1}
        else:
            class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    elif args.dataset.lower() == 'utkface_race':
        ngroups = 5
        if args.imbalance == 1:
            class_perc_dict = {0: 1, 1: 1, 2: args.minorityprop, 3: 1, 4: args.minorityprop}
        else:
            class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    elif args.dataset.lower() == 'celeba':
        is_celeba = True
        ngroups = 8
        class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}

    tmodel = load_model(args.tnet, args, ngroups) 
    smodel = load_model(args.net, args, ngroups)
    print(class_perc_dict)
    cifar10_train, cifar10_test, leftout_training_set, leftout_test_set = unbalance_dataset(
    args.dataset, class_perc_dict=class_perc_dict, seed=args.seed, data_dir = "/ReLU_Reduction_Fairness-main/DeepReduce/data", is_celeba = is_celeba)#"./data")
    print('loaded dataset')

    trainloader = DataLoader(cifar10_train, batch_size=args.batch, shuffle=True, num_workers=2)
    testloader = DataLoader(cifar10_test, batch_size=args.batch, shuffle=False, num_workers=2)
    print('dataloaders ready')
    #trainloader = reshape_and_create_loader(trainloader) #redundant for the new dataset.py
    #testloader = reshape_and_create_loader(testloader) #redundant for the new dataset.py

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger(__name__)
    print('trainingggg')
    train_DeepReduce(args,
                    trainloader,
                    testloader,
                    device,
                    logger,
                    tmodel,
                    smodel)





