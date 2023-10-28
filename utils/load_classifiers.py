from robustbench import load_model
from torchvision import models
import torch
import os
from clf_models.mnist import *
from clf_models.cifar10 import *
from edm.torch_utils import distributed as dist


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove 'module.0.' (idx_start=9) or 'module.' (idx_start=7) of dataparallel 
        name = k[idx_start:]  
        new_state_dict[name]=v

    return new_state_dict

def load_classifier(dataset, classifier_name, device):

    if dataset == 'ImageNet':
        if classifier_name == 'resnet18':
            dist.print0(f'using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif classifier_name == 'resnet50':
            dist.print0(f'using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif classifier_name == 'resnet152':
            dist.print0(f'using imagenet resnet152...')
            model = models.resnet152(pretrained=True).eval()
        elif classifier_name == 'resnet101':
            dist.print0(f'using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif classifier_name == 'wideresnet-50-2':
            dist.print0(f'using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        # elif classifier_name == 'deit-s':
        #     dist.print0('using imagenet deit-s...')
        #     model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')

    
    elif dataset == 'MNIST':
        if classifier_name == 'mnist-lenet':
            dist.print0('using mnist lenet...')
            model = MNISTLeNet()
            
            model_path = os.path.join('pretrained_models', 'mnist-lenet.tar')
            dist.print0(f"=> loading mnist-lenet checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded mnist-lenet checkpoint")
        
        elif classifier_name == 'mnist-lenet-raw-data':
            dist.print0('using mnist lenet raw data...')
            model = MNISTLeNet()
            
            model_path = os.path.join('pretrained_models', 'mnist-lenet-raw-data.tar')
            dist.print0(f"=> loading mnist-lenet-raw-data checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded mnist-lenet-raw-data checkpoint")
        
    
    elif dataset == 'CIFAR10' or dataset == 'CIFAR10-C':
        if classifier_name == 'wideresnet-28-10-ckpt':
            dist.print0('using cifar10 wideresnet-28-10-ckpt...')
            states_dict = torch.load(os.path.join('pretrained_models', 'cifar10-wrn-28-10.t7'), map_location=device)
            model = states_dict['net']
            
        elif classifier_name == 'wideresnet-28-10':
            dist.print0('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')  # pixel in [0, 1]

        elif classifier_name == 'wideresnet-70-16':
            dist.print0('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            model_path = os.path.join('pretrained_models', 'cifar10-wrn-70-16.pt')
            dist.print0(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            dist.print0(f"=> loaded wideresnet-70-16 checkpoint")

        elif classifier_name == 'wrn-70-16-dropout':
            dist.print0('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = os.path.join('pretrained_models', 'cifar10-wrn-70-16-dropout.pt')
            dist.print0(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded wrn-70-16-dropout checkpoint")

        elif classifier_name == 'resnet-50':
            dist.print0('using cifar10 resnet-50...')
            model = ResNet50()  # pixel in [0, 1]

            model_path = os.path.join('pretrained_models', 'cifar10-resnet-50.pt')
            dist.print0(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            dist.print0(f"=> loaded resnet-50 checkpoint")

        else:
            raise NotImplementedError(f'unknown {classifier_name}')

    elif dataset == 'CIFAR100' or dataset == 'CIFAR100-C':
        if classifier_name == 'wideresnet-28-10-ckpt':
            dist.print0('using cifar100 wideresnet-28-10-ckpt...')
            states_dict = torch.load(os.path.join('pretrained_models', 'cifar100-wrn-28-10.t7'), map_location=device)
            model = states_dict['net']
        
        elif classifier_name == 'wideresnet-70-16-ckpt':
            dist.print0('using cifar100 wideresnet-70-16-ckpt...')
            states_dict = torch.load(os.path.join('pretrained_models', 'cifar100-wrn-70-16.t7'), map_location=device)
            model = states_dict['net']
    
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    return model