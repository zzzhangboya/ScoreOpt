import sys
sys.path.append('./clf_models')
sys.path.append('./edm')

import os
import argparse
import click
import logging
import pickle
import random
from edm import dnnlib
from edm.torch_utils import distributed as dist
import numpy as np
import pandas as pd
import torch
from utils import *
from eval_transfer import classifier_attack_and_purif

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--log', default='logs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='defalt.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--data_seed', nargs='+', help='Random seed for data subsets')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--clf_net', type=str, default='wideresnet-28-10')
    parser.add_argument('--subset_size', type=int, default=64, help='Size of the fixed subset')    
    
    # Purify
    parser.add_argument('--purify_iter', type=int, default=1, help='Number of iterations for purify')
    parser.add_argument('--purify_model', type=str, default='opt', choices=['ve', 'vp', 'edm', 'opt', 'None'])
    parser.add_argument('--purify_method', type=str, default='None', choices=['x0', 'xt', 'None'])
    parser.add_argument('--total_steps', type=int, default=1, help='Number of total diffusion steps')
    parser.add_argument('--forward_steps', type=float, default=1, help='Number of forward diffusion timesteps/Noise Scale of forward Process')
    
    # Optimization
    parser.add_argument('--loss_lambda', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--init', action= "store_true")

    # Attack
    parser.add_argument('--att_method', type=str, default='clf_pgd', choices=['fgsm', 'clf_pgd', 'bpda', 'bpda_eot', 'pgd_eot'])
    parser.add_argument('--att_lp_norm', type=int, default=-1, choices=[-1,1,2])
    parser.add_argument('--att_eps', type=float, default=8/255., help='8/255. for Linf, 0.5 for L2')
    parser.add_argument('--att_step', type=int, default=40, help='Step number of pgd attacks')
    parser.add_argument('--att_n_iter', type=int, default=50, help='Iteration number of adaptive attacks')
    parser.add_argument('--att_alpha', type=float, default=2/255., help='One-step attack pixel scale')
    parser.add_argument('--eot_defense_reps', type=int, default=150, help='Number of EOT for adaptive attacks')
    parser.add_argument('--eot_attack_reps', type=int, default=15, help='Number of EOT for defenses')

    # edm
    parser.add_argument('--network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
    parser.add_argument('--sigma_min', help='Lowest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
    parser.add_argument('--sigma_max', help='Highest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))

    args = parser.parse_args()
    args.log = os.path.join(args.log, args.dataset, args.att_method, "l{}_{}x{}_it_{}".format(
            args.att_lp_norm,
            args.subset_size,
            len(args.data_seed),
            args.att_step
            ),
            "model_{}_method_{}_total_{}_forward_{}_pur_{}".format(
            args.purify_model,
            args.purify_method,
            args.total_steps,
            args.forward_steps,
            args.purify_iter
            ),
            "lr_{}_init_{}_lambda_{}_seed_{}".format(
            args.lr,
            args.init,
            args.loss_lambda,
            args.seed
            ))
    if not os.path.exists(args.log):
        os.makedirs(args.log, exist_ok=True)
        
    # set logger  
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'seed_{}.txt'.format(args.seed)))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
        
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    return args



def main():
    args = parse_args_and_config()
    dist.init()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
        
    if dist.get_rank() == 0:
        logging.info("Using device: {}".format(device))
        logging.info("Writing log file to {}".format(args.log))
        logging.info("Using args: {}".format(args))
    
    # dataset and pre-trained model
    val_dataloader = preprocess_datasets(args.dataset, False, args.batch_size, args.data_seed, args.subset_size, dist=True)
    
    clf = load_classifier(args.dataset, args.clf_net, args.device).to(args.device)
    clf.eval()
    trans_to_clf = get_transforms(args.dataset, args.clf_net)

    if args.dataset == 'CIFAR10':
        dist.print0(f"=> loading cifar10-diffusion checkpoint from '{args.network_pkl}'")
        with dnnlib.util.open_url(args.network_pkl, verbose=(dist.get_rank() == 0)) as f:
            score = pickle.load(f)['ema'].to(args.device)
    elif args.dataset == 'CIFAR100':
        dist.print0(f"=> loading cifar100-diffusion checkpoint from '{args.network_pkl}'")
        with dnnlib.util.open_url(args.network_pkl, verbose=(dist.get_rank() == 0)) as f:
            score = pickle.load(f)['ema'].to(args.device)
    
    torch.distributed.barrier()

    std_acc, rob_acc, purif_std_acc, purif_rob_acc, cnt \
        = classifier_attack_and_purif(args, val_dataloader, clf, trans_to_clf, score, None)
    t = torch.tensor([std_acc, rob_acc, purif_std_acc, purif_rob_acc, cnt], dtype=torch.int, device='cuda')
    
    torch.distributed.barrier()
    torch.distributed.all_reduce(t)
    t = t.cpu().numpy()
    total_count = t[-1]
    t = t / total_count
    
    if dist.get_rank() == 0:
        logging.info("Rank: 0")
        logging.info('count: {}'.format(cnt))
        logging.info('standard accuracy: {}'.format(std_acc))
        logging.info('robust accuracy: {}'.format(rob_acc))
        logging.info('purif standard accuracy: {}'.format(purif_std_acc))
        logging.info('purif robust accuracy: {}'.format(purif_rob_acc))
        
        logging.info("All gpus")
        logging.info('count: {}'.format(total_count))
        logging.info('standard accuracy: {}'.format(t[0]))
        logging.info('robust accuracy: {}'.format(t[1]))
        logging.info('purif standard accuracy: {}'.format(t[2]))
        logging.info('purif robust accuracy: {}'.format(t[3]))
        
        df=pd.DataFrame()
        new_row ={
                "dataset":args.dataset, 
                "att_method":args.att_method,
                "att_lp_norm":args.att_lp_norm,
                "subset_size":args.subset_size,
                "subset_number":len(args.data_seed),
                "att_step":args.att_step,
                "att_n_iter":args.att_n_iter,
                "att_eot_defense":args.eot_defense_reps,
                "att_eot_attack":args.eot_attack_reps,
                "purify_type":args.purify_model,
                "purify_method":args.purify_method,
                "total_steps":args.total_steps,
                "forward_steps":args.forward_steps,
                "purify_iter":args.purify_iter,
                "lr":args.lr,
                "init":args.init,
                "loss_lambda":args.loss_lambda,
                'seed':args.seed,
                "std_acc":t[0],
                "rob_acc":t[1],
                "purif_std_acc":t[2], 
                "purif_rob_acc":t[3]
        }
        df = df.append(new_row, ignore_index=True)
        df.to_csv(os.path.join("results","{}_{}_l{}_{}x{}_iter_{}_model_{}_method_{}_total_{}_forward_{}_pur_{}_seed_{}.csv").format(
                args.dataset, 
                args.att_method,
                args.att_lp_norm,
                args.subset_size,
                len(args.data_seed),
                args.att_step,
                args.purify_model,
                args.purify_method,
                args.total_steps,
                args.forward_steps,
                args.purify_iter,
                args.seed     
        ))
    
if __name__ == '__main__':
    main()


