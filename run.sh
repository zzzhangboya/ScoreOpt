torchrun --nproc_per_node=1 main_transfer.py --dataset 'CIFAR10' --batch_size 128 \
                        --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
                        --clf_net 'wideresnet-28-10-ckpt' --subset_size -1  --data_seed 0 --seed 1234 \
                        --purify_model 'opt' --forward_steps 0.25 --total_steps 1 --purify_iter 20 \
                        --att_method 'clf_pgd' --att_lp_norm -1 --att_eps 0.031373 --att_step 40 \
                        --lr 0.01 --purify_method 'x0'

# torchrun --nproc_per_node=1 main_transfer.py --dataset 'CIFAR10' --batch_size 128 \
#                         --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
#                         --clf_net 'wideresnet-28-10-ckpt' --subset_size -1 --data_seed 0 --seed 1234 \
#                         --purify_model 'opt' --forward_steps 0.25 --total_steps 1 --purify_iter 5 \
#                         --att_method 'clf_pgd' --att_lp_norm -1 --att_eps 0.031373 --att_step 40 \
#                         --lr 0.1 --purify_method 'xt'

# torchrun --nproc_per_node=1 main_bpda_eot.py --dataset 'CIFAR10' --batch_size 8 \
#                 --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
#                 --clf_net 'wideresnet-28-10-ckpt' --subset_size 512 --data_seed 0 --seed 1234\
#                 --purify_model 'opt' --forward_steps 0.5 --total_steps 1 --purify_iter 5\
#                 --att_method 'bpda_eot' --att_lp_norm -1 --att_eps 0.031373 --att_n_iter 50 --eot_attack_reps 15\
#                 --lr 0.1 --purify_method 'x0'

# torchrun --nproc_per_node=1 main_bpda_eot.py --dataset 'CIFAR10' --batch_size 8 \
#                 --network_pkl "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
#                 --clf_net 'wideresnet-28-10-ckpt' --subset_size 512 --data_seed 0 --seed 1234\
#                 --purify_model 'opt' --forward_steps 0.5 --total_steps 1 --purify_iter 20 \
#                 --att_method 'pgd_eot' --att_lp_norm -1 --att_eps 0.031373 --att_n_iter 20 --eot_attack_reps 20\
#                 --lr 0.1 --purify_method 'xt'