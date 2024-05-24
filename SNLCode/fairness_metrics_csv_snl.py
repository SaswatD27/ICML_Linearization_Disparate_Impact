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
import re
import csv
import ast
from tqdm import tqdm
import sys
from pyhessian import hessian
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_path = os.path.join(parent_dir, 'src')
sys.path.append(src_path)

from dataset import *
from torchvision import datasets

def get_hessian(model, test_loader, criterion, n_classes):
    hessian_norm_dict = {i: [] for i in range(n_classes)}
    for x_test, y_test in test_loader:
        y_test = y_test.squeeze().long()
        for i in range(n_classes):
            if len(y_test[y_test == i]) > 0:
                criterion = torch.nn.CrossEntropyLoss()
                mask = extract_for_group(x_test, y_test, i)
                hessian_comp = hessian(model, criterion, data=(x_test[mask], y_test[mask]), cuda=True)
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                hessian_norm_dict[i].append(top_eigenvalues[0])
    return hessian_norm_dict

def get_grad_norms(model, test_loader, criterion, n_classes, device):
    grad_norm_dict = {i: [] for i in range(n_classes)}
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        y_test = y_test.squeeze().long()
        y_pred = model(x_test)
        for i in range(n_classes):
            if len(y_test[y_test == i]) > 0:
                model.zero_grad()
                #print([w for w in model.parameters()])
                #sys.exit()
                group_loss = criterion(y_pred[y_test == i], y_test[y_test == i])
                group_loss.backward(retain_graph=True)
                sub_norm = torch.norm(torch.stack([torch.norm(w.grad) for w in model.parameters()])).item()
                grad_norm_dict[i].append(sub_norm)
    return grad_norm_dict

def get_dist_to_decision_boundary():
    pass

'''
def get_grad_norms(model, test_loader, criterion, n_classes, device):
    grad_norm_dict = {i: [] for i in range(n_classes)}
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_test = y_test.squeeze().long()
            y_pred = model(x_test)
            for i in range(n_classes):
                mask = (y_test == i)
                if torch.sum(mask) > 0:
                    model.zero_grad()
                    group_loss = criterion(y_pred[mask], y_test[mask])

                    # Use torch.autograd.grad to get gradients directly
                    grads = torch.autograd.grad(group_loss, model.parameters(), retain_graph=True)

                    # Calculate the norm of all gradients
                    sub_norm = torch.norm(torch.stack([torch.norm(g) for g in grads])).item()
                    grad_norm_dict[i].append(sub_norm)

                    # Clear gradients to reduce memory overhead
                    model.zero_grad()
    model.train()  # Set the model back to training mode
    return grad_norm_dict
'''

def count_relu(model, input_size):
  from torchinfo import summary as summary_
  from torch import nn
  layer_id = []
  count = []

  relu_instance_count = 0
  for m in model.modules():
    if isinstance(m, nn.ReLU):
      relu_instance_count += 1
  summary_list = summary_(model, input_size=input_size).summary_list
  # print(summary_(model, input_size=input_size))
  for i in range(len(summary_list)):
    # print(summary_list[i].get_layer_name(True, True))
    if 'relu' in summary_list[i].get_layer_name(True, True).lower():
      output_size = summary_list[i].output_size
      count.append(np.prod(output_size[1:]))
      layer_id.append(summary_list[i].get_layer_name(True, True))
  return layer_id, count

'''
def extract_for_group(inputs, targets, label):
    mask = (targets == label).nonzero().squeeze()
    inputs_extracted = inputs[mask]
    targets_extracted = targets[mask]
    return inputs_extracted, targets_extracted
'''
'''
def extract_for_group(inputs, targets, label):
    mask = (targets == label).nonzero().squeeze()
    return mask
'''

def extract_for_group(inputs, targets, label):
    mask = (targets == label).nonzero().squeeze()
    if torch.sum(targets == label) == 1:
        mask = mask.unsqueeze(dim = 0)
    return mask

def get_gender_indices_celeba(split, data_dir = "/ReLU_Reduction_Fairness/DeepReduce/data"):
    image_size = 32
    transform=transforms.Compose([transforms.Resize((image_size,image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    celeba_dataset = datasets.CelebA(root=data_dir, transform=transform, split=split, download=False)
    attributes = celeba_dataset.attr
    male_index = celeba_dataset.attr_names.index('Male')
    #print(f'Gender labels: {attributes[:, male_index]}')
    male_indices = attributes[:, male_index] == 1
    female_indices = attributes[:, male_index] == 0
    '''
    male_indices = torch.nonzero(attributes[:, male_index] == 1).squeeze()
    all_indices = torch.arange(len(celeba_dataset))
    female_indices = np.setdiff1d(all_indices.numpy(), male_indices.numpy())
    '''
    return [female_indices, male_indices]

def evaluate_loss_and_accuracy(model, dataloader, criterion, ngroups, device, is_celeba = False, split = 'train'):
    '''
    if is_celeba:
        print('Evaluating for CelebA')
        ngroups = 2
        gender_indices_list = get_gender_indices_celeba(split)
    '''
    sample_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    class_loss = torch.zeros(ngroups, dtype=torch.float, device=device)
    class_total = torch.zeros(ngroups, dtype=torch.float, device=device)
    class_correct = torch.zeros(ngroups, dtype=torch.float, device=device)
    
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        losses_per_sample = []
        differences_accumulated = {g: [] for g in range(ngroups)} 
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(dim=1).long()

            outputs = model(inputs)
            predictions.append(torch.argmax(outputs, dim=1).reshape(-1,1))
            targets.append(labels.reshape(-1,1))
            
            #loss = criterion(outputs, labels)
            
            loss_per_sample = sample_criterion(outputs, labels)
            losses_per_sample.append(loss_per_sample.reshape(-1,1))
            #print(f'loss dim: {loss.shape}')
            #print(f'loss_per_sample dim: {loss_per_sample.shape}')
            softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
            for g in range(ngroups):
                g_indices = (labels == g)  # Identify indices for group g

                # Filter outputs for the current group
                group_softmax_scores = softmax_scores[g_indices]

                # Compute top two scores
                if group_softmax_scores.size(0) > 0:  # Check if there are elements in the group
                    top_two_scores, _ = torch.topk(group_softmax_scores, 2, dim=1)
                    differences = top_two_scores[:, 0] - top_two_scores[:, 1]
                    differences_accumulated[g].append(differences)


        losses_per_sample = torch.vstack(losses_per_sample)
        predictions = torch.vstack(predictions)
        targets = torch.vstack(targets)
        dist_to_decision_boundary = {}
        #dist_to_decision_boundary = all_differences.mean().item()
        #print(f'predictions: {predictions}')
        #print(f'unique predictions: {np.unique(predictions.detach().cpu().numpy(), return_counts = True)}')
        #print(f'targets: {targets}')
        global_accuracy = (predictions == targets).sum()/float(len(predictions))
        class_accuracies = {}
        class_losses = {}
        class_num = {}
        dist_to_decision_boundary = {}
        for g in range(ngroups):
            #if is_celeba:
            #    g_indices = gender_indices_list[g]
            #else:
            g_indices = targets == g
            #print('PREDICTIONS', predictions[g_indices], 'TARGETS' ,targets[g_indices])
            #print('CORRECT: ', (predictions[g_indices] == targets[g_indices]).sum(),'\nWRONG: ',(predictions[g_indices] != targets[g_indices]).sum(),'\nTOTAL: ', float(sum(g_indices)))
            class_accuracy = (predictions[g_indices] == targets[g_indices]).sum()/float(sum(g_indices))
            class_loss = losses_per_sample[g_indices].mean()
            class_num[g] = sum(g_indices).item()#class_accuracy.item()
            class_accuracies[g] = class_accuracy.item()
            class_losses[g] = class_loss.mean().item()
            if differences_accumulated[g]:  # Check if there are elements for group g
                all_differences = torch.cat(differences_accumulated[g])
                dist_to_decision_boundary[g] = all_differences.mean().item()
            else:
                dist_to_decision_boundary[g] = None  # No data for this group
        print(f'Accuracies: {class_accuracies}')
        print(f'Losses: {class_losses}')
        print(f'Distance to Decision Boundary: {dist_to_decision_boundary}')
        print(global_accuracy.item())
        return class_accuracies, class_losses, class_num, global_accuracy.item(), dist_to_decision_boundary

def extract_params_from_path(checkpoint_path, model_path):
    # Get the relative path by removing the model_path prefix
    relative_path = os.path.relpath(checkpoint_path, model_path)
    print(relative_path)
    # Define the pattern for matching the desired directory structure
    #pattern = re.compile(r'culled([^/]+)/thinned_([^_]+)_alpha([^_]+)_seed([^/]+)/')
    pattern = re.compile(r'([^/]+)/seed_([^/]+)/snl_reduce_seed_([^/]+)/budget_([^/]+)/snl_best_checkpoint_([^/]+)_([^/]+)_([^/]+).pth.tar')
    # Try to match the pattern
    match = pattern.search(relative_path)
    #/resnet18_in/seed_1/snl_reduce_seed_1/budget_2.00/snl_best_checkpoint_resnet18_in_cifar10_9830.pth.tar
    if match:
        # Extract the matched groups
        model_name, split_seed_val, reduce_seed_val, perc_budget, _, _, n_relus = match.groups()
        
        return {
            'model': model_name,
            'split_seed': split_seed_val,
            'snl_reduce_seed': reduce_seed_val,
            'percent_budget': perc_budget,
            'n_relus': n_relus
        }
    else:
        print(f'{checkpoint_path} not valid')
        assert(1==0) #ad hoc

def extract_params_from_orig_path(checkpoint_path, model_path):
    # Get the relative path by removing the model_path prefix
    relative_path = os.path.relpath(checkpoint_path, model_path)
    print(relative_path)
    # Define the pattern for matching the desired directory structure
    #pattern = re.compile(r'culled([^/]+)/thinned_([^_]+)_alpha([^_]+)_seed([^/]+)/')
    pattern = re.compile(r'([^/]+)/seed_([^/]+)/best_checkpoint.pth.tar')
    # Try to match the pattern
    match = pattern.search(relative_path)
    #/resnet18_in/seed_1/snl_reduce_seed_1/budget_2.00/snl_best_checkpoint_resnet18_in_cifar10_9830.pth.tar
    if match:
        # Extract the matched groups
        model_name, split_seed_val = match.groups()

        if model_name == 'resnet18_in':
            n_relus = 491520
        
        return {
            'model': model_name,
            'split_seed': split_seed_val,
            #'snl_reduce_seed': reduce_seed_val,
            'percent_budget': 1.0,
            'n_relus': n_relus
        }
    else:
        print(f'{checkpoint_path} not valid')
        assert(1==0) #ad hoc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('dataset', type=str)#, choices=DATASETS)
    parser.add_argument('--arch', default='resnet18_in', type=str)#, choices=ARCHITECTURES)
    #parser.add_argument('outdir', type=str, help='folder to save model and training log)')
    #parser.add_argument('savedir', type=str, help='folder to load model')
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

    parser.add_argument('--saved_models_path', type=str, default='/ReLU_Reduction_Fairness/SNLCode/saved_models/')
    parser.add_argument('--original_models_path', type=str, default='/ReLU_Reduction_Fairness/SNLCode/saved_models/cifar10')
    parser.add_argument('--original_models_path_ignore', type=int, default=0) # 1 for True
    parser.add_argument('--grad_norm_ignore', type=int, default=0) # 1 for True
    parser.add_argument('--imbalance', type=int, default=0) # 1 for True
    parser.add_argument('--ngroups', type=int, default=5)
    parser.add_argument('--csv_file_path', type=str, default='/ReLU_Reduction_Fairness/SNLCode/snl_fairness.csv')
    args = parser.parse_args()

    #total_csv_list = [['#ReLUs', 'Group', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss']]
    total_csv_list = [['#ReLUs', 'Group', 'No. of Training Members', 'No. of Testing Members', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Hessian', 'Grad Norm', 'Train Decision Boundary Distance', 'Test Decision Boundary Distance', 'Global Train Accuracy', 'Global Test Accuracy']]

    checkpoint_paths = []
    model_path = args.saved_models_path
    for root, dirs, files in os.walk(model_path):
        for file in files:
            # Check if the file has a .pth.tar extension
            if file.endswith('.pth.tar'):
                # Create the full path to the checkpoint file
                checkpoint_file_path = os.path.join(root, file)
                # Append the path to the checkpoint_path list
                checkpoint_paths.append(checkpoint_file_path)
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
            class_perc_dict = {i:1 for i in range(10)}
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
    ngroups = args.ngroups
    groups = list(range(ngroups))
    
    train_set, test_set, leftout_training_set, leftout_test_set = unbalance_dataset(
args.dataset, class_perc_dict=class_perc_dict, seed=args.seed, data_dir = "/ReLU_Reduction_Fairness/DeepReduce/data", is_celeba = is_celeba)
    

    original_checkpoint_paths = []

    if args.original_models_path_ignore == 0:
        for root, dirs, files in os.walk(args.original_models_path):
            for file in files:
                # Check if the file has a .pth.tar extension
                if file.endswith('.pth.tar'):
                    # Create the full path to the checkpoint file
                    checkpoint_file_path = os.path.join(root, file)
                    # Append the path to the checkpoint_path list
                    original_checkpoint_paths.append(checkpoint_file_path)

    for checkpoint_path in original_checkpoint_paths:
        #print(checkpoint_path)
        param_dict = extract_params_from_orig_path(checkpoint_path, model_path)
        '''
        {
            'model': model_name,
            'split_seed': split_seed_val,
            'snl_reduce_seed': reduce_seed_val,
            'percent_budget': perc_budget,
            'n_relus': n_relus
        }
        '''
        print(param_dict)
        #args.net = param_dict['studentmodel']
        args.arch = param_dict['model']
        args.seed = 1#int(param_dict['snl_reduce_seed'])
        args.relu_budget = float(param_dict['percent_budget'])
        n_relus = int(param_dict['n_relus'])
        #args.seed = int(param_dict['seed'])
        checkpoint = torch.load(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_architecture(args.arch, args.dataset, device, args)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2)
        testloader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=2)

        if torch.cuda.is_available():
            criterionCls = torch.nn.CrossEntropyLoss().to(device)
            model = model.to(device)
        else:
            criterionCls = torch.nn.CrossEntropyLoss()
        
        #get accuracies and losses
        train_accs, train_losses, train_class_num, global_train_acc = evaluate_loss_and_accuracy(model, trainloader, criterionCls, ngroups, device, is_celeba, split = 'train')
        test_accs, test_losses, test_class_num, global_test_acc = evaluate_loss_and_accuracy(model, testloader, criterionCls, ngroups, device, is_celeba, split = 'test')
        hessians = get_hessian(model, testloader, criterionCls, ngroups)
        grad_norms = get_grad_norms(model, testloader, criterionCls, ngroups, device)
        #[['#ReLUs', 'Group', 'No. of Training Members', 'No. of Testing Members', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Global Train Accuracy', 'Global Test Accuracy']]

        csv_list = []
        for g in groups:
            csv_list.append([n_relus, g, train_class_num[g], test_class_num[g], train_accs[g], train_losses[g], test_accs[g], test_losses[g], hessians[g], grad_norms[g], global_train_acc, global_test_acc])
            print([n_relus, g, train_class_num[g], test_class_num[g], train_accs[g], train_losses[g], test_accs[g], test_losses[g], hessians[g], grad_norms[g], global_train_acc, global_test_acc]) #just because; redundant

        #print(csv_list)
        #----end loop here-----
        total_csv_list.extend(csv_list)
        #print(f'total list: {total_csv_list}')

    #print('CHECKPOINT PATHS: ',checkpoint_paths)
    #----begin loop here-----
    for checkpoint_path in tqdm(checkpoint_paths):
        #print(checkpoint_path)
        param_dict = extract_params_from_path(checkpoint_path, model_path)
        '''
        {
            'model': model_name,
            'split_seed': split_seed_val,
            'snl_reduce_seed': reduce_seed_val,
            'percent_budget': perc_budget,
            'n_relus': n_relus
        }
        '''
        print(param_dict)
        #args.net = param_dict['studentmodel']
        args.arch = param_dict['model']
        args.seed = int(param_dict['snl_reduce_seed'])
        args.relu_budget = float(param_dict['percent_budget'])
        n_relus = int(param_dict['n_relus'])
        #args.seed = int(param_dict['seed'])
        checkpoint = torch.load(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_architecture(args.arch, args.dataset, device, args)
        print(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        trainloader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2)
        testloader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=2)

        if torch.cuda.is_available():
            criterionCls = torch.nn.CrossEntropyLoss().to(device)
            model = model.to(device)
        else:
            criterionCls = torch.nn.CrossEntropyLoss()
        
        #get accuracies and losses
        train_accs, train_losses, train_class_num, global_train_acc, dist_to_decision_boundary_train = evaluate_loss_and_accuracy(model, trainloader, criterionCls, ngroups, device, is_celeba, split = 'train')
        test_accs, test_losses, test_class_num, global_test_acc, dist_to_decision_boundary_test = evaluate_loss_and_accuracy(model, testloader, criterionCls, ngroups, device, is_celeba, split = 'test')
        #[['#ReLUs', 'Group', 'No. of Training Members', 'No. of Testing Members', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Global Train Accuracy', 'Global Test Accuracy']]
        hessians = get_hessian(model, testloader, criterionCls, ngroups)
        if args.grad_norm_ignore == 0:
            grad_norms = get_grad_norms(model, testloader, criterionCls, ngroups, device)
        else:
            grad_norms = hessians #MEANINGLESS; IGNORE THIS COLUMN IN THE CSV

        csv_list = []
        for g in groups:
            csv_list.append([n_relus, g, train_class_num[g], test_class_num[g], train_accs[g], train_losses[g], test_accs[g], test_losses[g], np.max(hessians[g]), np.mean(grad_norms[g]), dist_to_decision_boundary_train[g], dist_to_decision_boundary_test[g], global_train_acc, global_test_acc])
            print([n_relus, g, train_class_num[g], test_class_num[g], train_accs[g], train_losses[g], test_accs[g], test_losses[g], np.max(hessians[g]), np.mean(grad_norms[g]), dist_to_decision_boundary_train[g], dist_to_decision_boundary_test[g], global_train_acc, global_test_acc]) #just because; redundant

        #print(csv_list)
        #----end loop here-----
        total_csv_list.extend(csv_list)

    #convert to csv
    with open(args.csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Writing each row in the list of lists to the CSV file
        for row in total_csv_list:
            csv_writer.writerow(row)
                
    print(f"CSV file '{args.csv_file_path}' has been created.")