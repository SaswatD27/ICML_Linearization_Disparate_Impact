import os
import gdown
import torch
import torchvision
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
from torchvision import transforms
#from utils import set_all_random_seeds
from sklearn.model_selection import train_test_split
import random
import numpy as np
import gdown
import deeplake
import sys
import itertools

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STDDEV = (0.2023, 0.1994, 0.2010)

def set_all_random_seeds(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract_data_targets(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    data_ = []
    targets_ = []
    with torch.no_grad():
        for _, (data, targets) in enumerate(tqdm(dataloader, desc="Extracting data and targets")): #enumerate(dataloader):
            data_.append(data)
            targets_.append(targets.reshape(-1, 1))
    data = torch.vstack(data_)
    print('data stacked')
    targets = torch.vstack(targets_)
    print('targets stacked')
    return data, targets

def extract_data_targets_celeba(dataset, batch_size, data_dir, split, target_attributes):
    data_ = []
    final_labels_ = []
    
    # Generate all possible combinations of binary attributes
    binary_combinations = list(itertools.product([0, 1], repeat=len(target_attributes)))

    # Find the indices of the target attributes in the list of attributes
    target_attr_indices = [dataset.attr_names.index(attr) if attr in dataset.attr_names else None for attr in target_attributes]

    # Create a dictionary to map binary combinations to final labels (0 to 7)
    binary_combination_to_label = {combo: i for i, combo in enumerate(binary_combinations)}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Extracting data and targets")):
            data_.append(data)

            # Extract the binary attributes for each batch using the batch indices
            binary_attributes_batch = torch.tensor(dataset.attr[batch_idx * batch_size:(batch_idx + 1) * batch_size, target_attr_indices], dtype=torch.int)

            # Convert binary attributes to tuples and map to final labels
            final_labels_batch = torch.tensor([binary_combination_to_label[tuple(attr.tolist())] for attr in binary_attributes_batch], dtype=torch.int).reshape(-1, 1)
            final_labels_.append(final_labels_batch)

    data = torch.vstack(data_)
    final_labels = torch.vstack(final_labels_)
    return data, final_labels, binary_combination_to_label

def unbalance_dataset(dataset_name: str, class_perc_dict: dict, seed: int, data_dir: str, transform=None, is_celeba=False):#, celeba_target = 'Male'):
    set_all_random_seeds(seed)

    num_classes = get_num_classes(dataset_name)
    print('Step 0 complete')

    assert len(class_perc_dict.keys(
    )) == num_classes, f"Number of classes in dataset {dataset_name} is {num_classes} but {len(class_perc_dict.keys())} class percentages were provided"

    assert all([0 < class_perc_dict[key] <= 1 for key in class_perc_dict.keys(
    )]), f"All values in class_perc_dict must be between larger than 0 and less or equal to 1"

    training_set, test_set = get_dataset(
        dataset_name, data_dir, transform=transform, seed = seed)
    print('Step 1 complete')
    if is_celeba:
        print('Running for CelebA')
        target_attributes = ['Male', 'Blond_Hair']#['No_Beard','Heavy_Makeup','Pale_Skin']#['Narrow_Eyes', 'Brown_Hair', 'Heavy_Makeup']
        train_data, train_labels, binary_combination_to_label = extract_data_targets_celeba(
            training_set, batch_size=1000, data_dir = f"{data_dir}/celeba", split = "train", target_attributes = target_attributes)#, target = celeba_target)
        test_data, test_labels, binary_combination_to_label = extract_data_targets_celeba(test_set, batch_size=1000, data_dir = f"{data_dir}/celeba", split = "test", target_attributes = target_attributes)#, target = celeba_target)
        print(f'Label Dictionary: {binary_combination_to_label}')
    else:
        train_data, train_labels = extract_data_targets(
            training_set, batch_size=1000)
        test_data, test_labels = extract_data_targets(test_set, batch_size=1000)

    print('Step 2 complete')


    new_train_data = []
    new_train_labels = []
    leftout_train_data = []
    leftout_train_labels = []
    new_test_data = []
    new_test_labels = []
    leftout_test_data = []
    leftout_test_labels = []

    for class_id, perc in class_perc_dict.items():
        # get all data and labels for this class
        # print(np.shape(train_labels.reshape(-1) == class_id))
        class_train_data = train_data[train_labels.reshape(-1) == class_id]
        class_train_labels = train_labels[train_labels.reshape(-1) == class_id]
        num_samples_to_keep = int(len(class_train_labels) * perc)
        random_indices = torch.randperm(len(class_train_labels))
        new_train_data.append(
            class_train_data[random_indices[:num_samples_to_keep]])
        new_train_labels.append(
            class_train_labels[random_indices[:num_samples_to_keep]].reshape(-1, 1))
        leftout_train_data.append(
            class_train_data[random_indices[num_samples_to_keep:]])
        leftout_train_labels.append(
            class_train_labels[random_indices[num_samples_to_keep:]].reshape(-1, 1))

        class_test_data = test_data[test_labels.reshape(-1) == class_id]
        class_test_labels = test_labels[test_labels.reshape(-1) == class_id]
        num_samples_to_keep = int(len(class_test_labels) * perc)
        random_indices = torch.randperm(len(class_test_labels))
        new_test_data.append(
            class_test_data[random_indices[:num_samples_to_keep]])
        new_test_labels.append(
            class_test_labels[random_indices[:num_samples_to_keep]].reshape(-1, 1))
        leftout_test_data.append(
            class_test_data[random_indices[num_samples_to_keep:]])
        leftout_test_labels.append(
            class_test_labels[random_indices[num_samples_to_keep:]].reshape(-1, 1))

    new_train_data = torch.vstack(new_train_data)
    new_train_labels = torch.vstack(new_train_labels)
    leftout_train_data = torch.vstack(leftout_train_data)
    leftout_train_labels = torch.vstack(leftout_train_labels)
    new_test_data = torch.vstack(new_test_data)
    new_test_labels = torch.vstack(new_test_labels)
    leftout_test_data = torch.vstack(leftout_test_data)
    leftout_test_labels = torch.vstack(leftout_test_labels)

    training_set = torch.utils.data.TensorDataset(
        new_train_data, new_train_labels)
    print('Train Labels: ',np.unique(new_train_labels.detach().cpu().numpy(), return_counts = True))
    leftout_training_set = torch.utils.data.TensorDataset(
        leftout_train_data, leftout_train_labels)
    test_set = torch.utils.data.TensorDataset(new_test_data, new_test_labels)
    print('Test Labels: ',np.unique(new_test_labels.detach().cpu().numpy(), return_counts = True))
    leftout_test_set = torch.utils.data.TensorDataset(
        leftout_test_data, leftout_test_labels)

    return training_set, test_set, leftout_training_set, leftout_test_set


def get_dataset(dataset_name: str, data_dir: str = "/ReLU_Reduction_Fairness-main/DeepReduce/data", seed = 42, transform=None):
    # lowercase the dataset name
    #print('Hello1')
    if dataset_name.lower() == 'cifar10':
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
            ])
        training_set = torchvision.datasets.CIFAR10(
            root=data_dir, download=True, transform=transform, train=True)
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, download=True, transform=transform, train=False)
        return training_set, test_set
    
    if dataset_name.lower() == 'svhn':
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
            ])
        training_set = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform)
        return training_set, test_set
    
    if dataset_name.lower() == 'cifar100':
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
            ])
        training_set = torchvision.datasets.CIFAR100(
            root=data_dir, download=True, transform=transform, train=True)
        test_set = torchvision.datasets.CIFAR100(
            root=data_dir, download=True, transform=transform, train=False)
        return training_set, test_set

    if dataset_name.lower() == 'celeba':
        if transform is None:
            # Spatial size of training images, images are resized to this size.
            image_size = 32

            transform=transforms.Compose([
                                        transforms.Resize((image_size,image_size)),
                                        #transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        #try:
        print(data_dir)
        training_set = torchvision.datasets.CelebA(
            root=data_dir, download=True, transform=transform, split='train', target_type = 'attr')
        test_set = torchvision.datasets.CelebA(
            root=data_dir, download=True, transform=transform, split='test', target_type = 'attr')
        #except:
        #    training_set = deeplake.load("hub://activeloop/celeb-a-train")
        #    test_set = deeplake.load("hub://activeloop/celeb-a-test")
        return training_set, test_set
    
    if dataset_name.lower() == 'celeba_unbalanced':
        if transform is None:
            # Spatial size of training images, images are resized to this size.
            image_size = 32 #64

            transform=transforms.Compose([
                                        transforms.Resize((image_size,image_size)),
                                        #transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #try:
        print(data_dir)
        training_set = torchvision.datasets.CelebA(
            root=data_dir, download=True, transform=transform, split='train', target_type = 'attr')
        test_set = torchvision.datasets.CelebA(
            root=data_dir, download=True, transform=transform, split='test', target_type = 'attr')
        #except:
        #    training_set = deeplake.load("hub://activeloop/celeb-a-train")
        #    test_set = deeplake.load("hub://activeloop/celeb-a-test")
        return training_set, test_set

    elif dataset_name.lower() == 'utkface_age':
        #print('Hello2')
        training_set = UTKFACEDataset_age(
            root=data_dir, transform=transform, train=True, split_seed = seed)
        #print('Hello3')
        test_set = UTKFACEDataset_age(
            root=data_dir, transform=transform, train=False, split_seed = seed)
        #print('Hello4')
        return training_set, test_set

    elif dataset_name.lower() == 'utkface_race':
        #print('Hello2')
        training_set = UTKFACEDataset_race(
            root=data_dir, transform=transform, train=True, split_seed = seed)
        #print('Hello3')
        test_set = UTKFACEDataset_race(
            root=data_dir, transform=transform, train=False, split_seed = seed)
        #print('Hello4')
        return training_set, test_set

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def get_num_classes(dataset_name: str):
    if dataset_name.lower() == 'cifar10':
        return 10
    if dataset_name.lower() == 'svhn':
        return 10
    if dataset_name.lower() == 'cifar100':
        return 100
    elif dataset_name.lower() == 'utkface_age':
        return 5
    elif dataset_name.lower() == 'utkface_race':
        return 5
    elif dataset_name.lower() == 'celeba':
        return 4
    elif dataset_name.lower() == 'celeba_unbalanced':
        return 2
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


class UTKFACEDataset_age(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, split_seed=42):
        # Mean and Std for ImageNet
        mean=[0.485, 0.456, 0.406] # ImageNet
        std=[0.229, 0.224, 0.225] # ImageNet

        # Define the Transforms
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        #print('Hola0')
        self.data_dir = root
        #self.transform = transform
        self.markerdata = "UTKFace"
        self.markertarget = "age"
        self.marker = f"{self.markerdata}_{self.markertarget}"
        #print('Hola1')
        utkFace_dir = os.path.join(root, f"{self.marker}")
        print('UTKFace Dir: ',utkFace_dir)
        if not os.path.exists(utkFace_dir):
            os.makedirs(root, exist_ok=True)
            self._download()
        #print('Hola2')
        self.image_paths, self.labels = self.load_data()
        #print('Hola3')
        tr_f, ts_f, tr_l, ts_l = train_test_split(
            self.image_paths,
            self.labels,
            test_size=0.2,
            random_state=split_seed,
            stratify=self.labels,
        )
        #print('Hola4')
        if train == True:
            self.image_paths = tr_f
            self.labels = tr_l
            data = []
            targets = []
            #print('Hola5')
            for idx in trange(len(self.image_paths)):
                image, label_tensor = self.__getitem__(idx)
                data.append(image)
                targets.append(label_tensor)

            self.data = torch.stack(data)
            # permute the data tensor to match the format expected by pytorch
            self.targets = torch.stack(targets)
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            c_dict = dict(zip(unique_labels, counts))
            print("Unique labels with their counts: ", c_dict)

        elif train == False:
            #print('Hola5.2')
            self.image_paths = ts_f
            self.labels = ts_l
            data = []
            targets = []
            for idx in trange(len(self.image_paths)):
                image, label_tensor = self.__getitem__(idx)
                data.append(image)
                targets.append(label_tensor)

            self.data = torch.stack(data)
            self.targets = torch.stack(targets)

    def load_data(self):
        image_paths = []
        labels = []
        for filename in os.listdir(os.path.join(self.data_dir, f"{self.marker}")):
            if filename.endswith(".jpg"):
                image_path = os.path.join(
                    os.path.join(self.data_dir, f"{self.marker}", filename)
                )
                image_paths.append(image_path)
                label = self.get_label_from_filename(filename)
                labels.append(label)
        return image_paths, labels

    def get_label_from_filename(self, filename):
        filename_parts = filename.split(".")[0].split("_")
        age = int(filename_parts[0])
        if 0 <= age <= 1:
            label = 0
        elif 2 <= age <= 12:
            label = 1
        elif 13 <= age <= 17:
            label = 2
        elif 18 <= age <= 64:
            label = 3
        else:
            label = 4
        return label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        #if self.transform is not None:
        image_tensor = self.transform(image)

        #else:
        #    image_tensor = img_to_tensor(image)
        label_tensor = torch.tensor(label)
        image.close()
        return image_tensor, label_tensor

    def _download(self):
        current_dir = os.getcwd()
        os.chdir(self.data_dir)
        url = "https://hydranets-data.s3.eu-west-3.amazonaws.com/UTKFace.zip"#"https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk"
        '''
        gdown.download(url, f"{self.markerdata}.tar.gz", quiet=False)
        os.system(f"tar -xvf {self.markerdata}.tar.gz")
        '''
        os.system(f"wget {url}")
        os.system("unzip UTKFace.zip")
        os.system(f"mv {self.markerdata} {self.marker}")
        #os.remove(f"{self.markerdata}.tar.gz")
        os.chdir(current_dir)

class UTKFACEDataset_race(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True, split_seed=42):
        # Mean and Std for ImageNet
        mean=[0.485, 0.456, 0.406] # ImageNet
        std=[0.229, 0.224, 0.225] # ImageNet

        # Define the Transforms
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        #print('Hola0')
        self.data_dir = root
        #self.transform = transform
        self.markerdata = "UTKFace"
        self.markertarget = "race"
        self.marker = f"{self.markerdata}_{self.markertarget}"
        #print('Hola1')
        utkFace_dir = os.path.join(root, f"{self.marker}")
        print('UTKFace Dir: ',utkFace_dir)
        if not os.path.exists(utkFace_dir):
            os.makedirs(root, exist_ok=True)
            self._download()
        #print('Hola2')
        self.image_paths, self.labels = self.load_data()
        #print({i:sum(self.labels==i) for i in range(5)})
        #print('Hola3')
        tr_f, ts_f, tr_l, ts_l = train_test_split(
            self.image_paths,
            self.labels,
            test_size=0.2,
            random_state=split_seed,
            stratify=self.labels,
        )
        #print('Hola4')
        if train == True:
            self.image_paths = tr_f
            self.labels = tr_l
            data = []
            targets = []
            #print('Hola5')
            for idx in trange(len(self.image_paths)):
                image, label_tensor = self.__getitem__(idx)
                data.append(image)
                targets.append(label_tensor)

            self.data = torch.stack(data)
            # permute the data tensor to match the format expected by pytorch
            self.targets = torch.stack(targets)
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            c_dict = dict(zip(unique_labels, counts))
            print("Unique labels with their counts: ", c_dict)

        elif train == False:
            #print('Hola5.2')
            self.image_paths = ts_f
            self.labels = ts_l
            data = []
            targets = []
            for idx in trange(len(self.image_paths)):
                image, label_tensor = self.__getitem__(idx)
                data.append(image)
                targets.append(label_tensor)

            self.data = torch.stack(data)
            self.targets = torch.stack(targets)

    def load_data(self):
        image_paths = []
        labels = []
        for filename in os.listdir(os.path.join(self.data_dir, f"{self.marker}")):
            if filename.endswith(".jpg"):
                image_path = os.path.join(
                    os.path.join(self.data_dir, f"{self.marker}", filename)
                )
                label = self.get_label_from_filename(filename)
                if label not in list(range(5)):
                    print(f'{label} is weird; datapoint ignored.')
                    continue
                image_paths.append(image_path)
                labels.append(label)
        return image_paths, labels

    def get_label_from_filename(self, filename):
        filename_parts = filename.split(".")[0].split("_")
        label = int(filename_parts[2])
        if label not in range(0, 5):
            print(f"\nThis filename is: {filename} and the label is: {label}")
        return label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        #if self.transform is not None:
        image_tensor = self.transform(image)

        #else:
        #    image_tensor = img_to_tensor(image)
        label_tensor = torch.tensor(label)
        image.close()
        return image_tensor, label_tensor

    def _download(self):
        current_dir = os.getcwd()
        os.chdir(self.data_dir)
        url = "https://hydranets-data.s3.eu-west-3.amazonaws.com/UTKFace.zip"#"https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk"
        '''
        gdown.download(url, f"{self.markerdata}.tar.gz", quiet=False)
        os.system(f"tar -xvf {self.markerdata}.tar.gz")
        '''
        os.system(f"wget {url}")
        os.system("unzip UTKFace.zip")
        os.system(f"mv {self.markerdata} {self.marker}")
        #os.remove(f"{self.markerdata}.tar.gz")
        os.chdir(current_dir)


def img_to_tensor(img):
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)
