import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from models.utils import load_model
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
import os
import sys
import re
import csv
import ast
from tqdm import tqdm
from train_DeepReduce import accuracy
import numpy as np
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
'''
def count_relu_units(model):
    relu_units = 0

    # Iterate through all modules in the model
    for module in model.modules():
        # Check if the module is an instance of ReLU
        if isinstance(module, nn.ReLU):
            relu_units += 1

    return relu_units
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

def extract_for_group(inputs, targets, label):
    mask = (targets == label).nonzero().squeeze()
    inputs_extracted = inputs[mask]
    targets_extracted = targets[mask]
    return inputs_extracted, targets_extracted

def evaluate_loss_and_accuracy(model, dataloader, criterion, ngroups, device):
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
        for inputs, labels in dataloader:
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


        losses_per_sample = torch.vstack(losses_per_sample)
        predictions = torch.vstack(predictions)
        targets = torch.vstack(targets)
        #print(f'predictions: {predictions}')
        #print(f'unique predictions: {np.unique(predictions.detach().cpu().numpy(), return_counts = True)}')
        #print(f'targets: {targets}')
        global_accuracy = (predictions == targets).sum()/float(len(predictions))
        class_accuracies = {}
        class_losses = {}
        class_num = {}
        for g in range(ngroups):
            g_indices = targets == g
            class_accuracy = (predictions[g_indices] == targets[g_indices]).sum()/float(sum(g_indices))
            class_loss = losses_per_sample[g_indices].mean()
            class_num[g] = sum(g_indices).item()#class_accuracy.item()
            class_accuracies[g] = class_accuracy.item()
            class_losses[g] = class_loss.mean().item()
        print(class_accuracies, class_losses)
        print(global_accuracy.item())
        return class_accuracies, class_losses, class_num, global_accuracy.item()

def extract_params_from_path(checkpoint_path, model_path):
    # Get the relative path by removing the model_path prefix
    relative_path = os.path.relpath(checkpoint_path, model_path)
    print(relative_path)
    # Define the pattern for matching the desired directory structure
    pattern = re.compile(r'culled([^/]+)/thinned_([^_]+)_alpha([^_]+)_seed([^/]+)/')
    
    # Try to match the pattern
    match = pattern.search(relative_path)
    
    if match:
        # Extract the matched groups
        culledmask, thinned, alpha, seed = match.groups()
        
        return {
            'culledmask': culledmask,
            'thinned': thinned,
            'alpha': alpha,
            'seed': seed
        }
    else:
        print(f'{checkpoint_path} not valid')
        assert(1==0) #ad hoc

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--tnet', type=str, default='resnet18_deepreduce_teacher_cifar10')
    args.add_argument('--net', type=str, default='resnet18_deepreduce_student_cifar10')
    args.add_argument('--dataset', type=str, default='cifar10') #utkface_age, utkface_race
    args.add_argument('--culled_mask', type=int, default=1)# or 14 or 12 #[False, True, True, True])
    args.add_argument('--thinned', type=int, default=0)
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

    args.add_argument('--ngroups', type=int, default=10)
    args.add_argument('--saved_models_path', type=str, default='/DeepReduce/saved_models/')
    args.add_argument('--csv_file_path', type=str, default='/deepreduce_fairness.csv')
    args = args.parse_args()

    ngroups = args.ngroups
    groups = list(range(ngroups))

    #total_csv_list = [['#ReLUs', 'Group', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss']]
    total_csv_list = [['#ReLUs', 'Group', 'No. of Training Members', 'No. of Testing Members', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Alphas', 'Thinned', 'Global Train Accuracy', 'Global Test Accuracy']]

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
        ngroups = 10
        class_perc_dict = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}
    if args.dataset.lower() == 'cifar100':
        ngroups = 100
        class_perc_dict = {i: 0.9 for i in range(50)}
        class_perc_dict.update({i: 0.3 for i in range(50, 100)})
    elif args.dataset.lower() == 'utkface_age':
        ngroups = 5
        class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    elif args.dataset.lower() == 'utkface_race':
        ngroups = 5
        class_perc_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    elif args.dataset.lower() == 'celeba':
        ngroups = 2
        is_celeba = True
        class_perc_dict = {0: 1, 1: 1}
    #add to csv and then continue loop over other pth.tar
    #cifar10_train, cifar10_test, leftout_training_set, leftout_test_set = unbalance_dataset('CIFAR10', class_perc_dict={0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1}, seed=args.seed, data_dir = "./data")
    
    cifar10_train, cifar10_test, leftout_training_set, leftout_test_set = unbalance_dataset(
args.dataset, class_perc_dict=class_perc_dict, seed=args.seed, data_dir = "./data", is_celeba=is_celeba)
    
    #print('CHECKPOINT PATHS: ',checkpoint_paths)
    #----begin loop here-----
    for checkpoint_path in tqdm(checkpoint_paths):
        #print(checkpoint_path)
        param_dict = extract_params_from_path(checkpoint_path, model_path)
        print(param_dict)
        #args.net = param_dict['studentmodel']
        args.culled_mask = int(param_dict['culledmask'])
        args.thinned = ast.literal_eval(param_dict['thinned'])
        args.alpha = float(param_dict['alpha'])
        args.seed = int(param_dict['seed'])
        checkpoint = torch.load(checkpoint_path)
        print(f'\n NET: {args.net, type(args.net)}\n')
        print(f'\n CULLED_MASK: {args.culled_mask, type(args.culled_mask)}\n')
        print(f'\n THINNED: {args.thinned, type(args.thinned)}\n')
        print(f'\n ALPHA: {args.alpha, type(args.alpha)}\n')
        print(f'\n SEED: {args.seed, type(args.seed)}\n')
        smodel = load_model(args.net, args, ngroups) #this is the model we need

        smodel.load_state_dict(checkpoint['model'])

        trainloader = DataLoader(cifar10_train, batch_size=args.batch, shuffle=True, num_workers=2)
        testloader = DataLoader(cifar10_test, batch_size=args.batch, shuffle=False, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            criterionCls = torch.nn.CrossEntropyLoss().to(device)
            smodel = smodel.to(device)
        else:
            criterionCls = torch.nn.CrossEntropyLoss()

        #count relus
        _, n_relus = count_relu(smodel, (1,3,32,32))
        n_relus = sum(n_relus)
        
        #get accuracies and losses
        train_accs, train_losses, train_class_num, global_train_acc = evaluate_loss_and_accuracy(smodel, trainloader, criterionCls, ngroups, device)
        test_accs, test_losses, test_class_num, global_test_acc = evaluate_loss_and_accuracy(smodel, testloader, criterionCls, ngroups, device)
        #[['#ReLUs', 'Group', 'No. of Training Members', 'No. of Testing Members', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Alphas', 'Thinned', 'Global Accuracy']]
        #return n_relus, train_accs, train_losses, test_accs, test_losses

        csv_list = []
        for g in groups:
            csv_list.append([n_relus, g, train_class_num[g], test_class_num[g], train_accs[g], train_losses[g], test_accs[g], test_losses[g], args.alpha, args.thinned, global_train_acc, global_test_acc])
            print([n_relus, g, train_class_num[g], test_class_num[g], train_accs[g], train_losses[g], test_accs[g], test_losses[g],  global_train_acc, global_test_acc]) #just because; redundant

        #print(csv_list)
        #----end loop here-----
        total_csv_list.extend(csv_list)
        #print(f'total list: {total_csv_list}')
    #convert to csv
    with open(args.csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Writing each row in the list of lists to the CSV file
        for row in total_csv_list:
            csv_writer.writerow(row)
                
    print(f"CSV file '{args.csv_file_path}' has been created.")

'''
            total_loss += loss.item() #* inputs.size(0)
            total_samples += inputs.size(0)

            class_acc, _ = accuracy(outputs, labels, topk=(1, 5))

            for i in range(len(labels)):
                outputs_g, labels_g = extract_for_group(outputs_g, labels_g)
                class_loss = criterion(outputs_g, labels_g)
                label = labels[i]
                class_loss[label] += class_loss.item()
                class_total[label] += 1
                class_accuracy[label] += class_acc#correct[i].item()
            
        #print(f'class_loss: {class_loss}')
        #print(f'class_correct: {class_correct}')
        #print(f'class_total: {class_total}')
        #overall_loss = total_loss / total_samples
        #overall_accuracy = #class_correct.sum().item() / class_total.sum().item()
        average_accuracy_per_class = class_accuracy / len(dataloader)#class_total
        average_loss_per_class = class_loss / len(dataloader)
        print(f'average_loss_per_class: {average_loss_per_class}')

    model.train()  # Set the model back to training mode

    return average_loss_per_class.tolist(), average_accuracy_per_class.tolist()
    '''