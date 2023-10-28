import torch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageNet, ImageFolder 
from torch.utils.data import Subset, TensorDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import math

data_dir='./datasets'

def foolbox_preprocess(args):
    if args.dataset in ["CIFAR10", "CIFAR10-C"] and args.clf_net in ['wideresnet-28-10-ckpt']:
        return dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    elif args.dataset in ["CIFAR100"]:
        return dict(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], axis=-3)
    elif args.dataset in ["ImageNet"]:
        return dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    elif args.dataset in ["MNIST"]:
        return dict(mean=[0.1307], std=[0.3081], axis=-3)
        # return dict()
    else:
        return dict()

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def preprocess_datasets(dataset, train, batch_size, data_seed, subset_size, dist=False, distortion_name=None, severity=None):
    '''
    dataset: datasets "MNIST", "FashionMNIST", "CIFAR10", "CIFAR10-C", "CIFAR100", "ImageNet","ImageNet-C"
    '''

    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(data_dir, "MNIST")
        val_dataset = MNIST(path, train=train, download=True, transform=transform)
        if subset_size > 0:
            partition_idx = []
            for seed in data_seed:
                partition_idx.extend(np.random.RandomState(int(seed)).choice(len(val_dataset), subset_size, replace=False))
            val_dataset = Subset(val_dataset, np.array(partition_idx))
    elif dataset=="CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(data_dir, "CIFAR10")
        val_dataset = CIFAR10(path, train=train, download=True, transform=transform)
        if subset_size > 0:
            partition_idx = []
            for seed in data_seed:
                partition_idx.extend(np.random.RandomState(int(seed)).choice(len(val_dataset), subset_size, replace=False))
            val_dataset = Subset(val_dataset, np.array(partition_idx))
    elif dataset=="ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()        
        ])
        path = os.path.join(data_dir, "ImageNet")
        val_dataset = ImageNet(path, split='val', transform=transform)
        if subset_size > 0:
            partition_idx = []
            for seed in data_seed:
                partition_idx.extend(np.random.RandomState(int(seed)).choice(len(val_dataset), subset_size, replace=False))
            val_dataset = Subset(val_dataset, partition_idx)
    elif dataset=="FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(data_dir, "FashionMNIST")
        val_dataset = FashionMNIST(path, train=train, download=True, transform=transform)
    elif dataset=="CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(data_dir, "CIFAR100")
        val_dataset = CIFAR100(path, train=train, download=True, transform=transform)
        if subset_size > 0:
            partition_idx = []
            for seed in data_seed:
                partition_idx.extend(np.random.RandomState(int(seed)).choice(len(val_dataset), subset_size, replace=False))
            val_dataset = Subset(val_dataset, np.array(partition_idx))
    elif dataset=="CIFAR10-C":
        dataloader = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(data_dir, "CIFAR-10-C")
        file_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', \
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        label_path = os.path.join(path, "labels.npy")
        lb_file = np.load(label_path) # Size [50000]
        np_y = lb_file[0:10000]
        for i in range(len(file_list)):
            sub_dataloader = []
            np_x = np.load(os.path.join(path, file_list[i]+".npy"))
            np_x = np.transpose(np_x, (0,3,1,2))
            for j in range(5):
                tensor_x = torch.Tensor(np_x[j*10000:(j+1)*10000])
                tensor_y = torch.Tensor(np_y)
                val_dataset = TensorDataset(tensor_x, tensor_y)
                sub_dataloader.append(DataLoader(val_dataset, batch_size=batch_size, num_workers=4))
            dataloader.append(sub_dataloader)
        return dataloader
    elif dataset=="ImageNet-C":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        path = os.path.join(data_dir, "ImageNet-C")
        # file_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', \
        #     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        valdir = path +'/' + distortion_name + '/' + str(severity)
        dataloader = torch.utils.data.DataLoader(
            ImageFolder(valdir, transform),
            batch_size=batch_size, num_workers=4, pin_memory=True)
        return dataloader
    
    if not dist:
        if dataset == "ImageNet":
            dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
        else:
            n_sample = len(val_dataset)
            print(f'n_sample: {n_sample}')
            dataloader = DataLoader(val_dataset, batch_size=n_sample, num_workers=4, shuffle=False, pin_memory=True)
        
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, sampler=dist_sampler)
    
    return dataloader
    
