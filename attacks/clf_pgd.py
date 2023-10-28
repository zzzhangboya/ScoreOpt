import foolbox
from foolbox.criteria import TargetedMisclassification
from utils.preprocess_datasets import foolbox_preprocess

def clf_pgd(x, y, diffusion, score, network_clf, args):
    fmodel = foolbox.PyTorchModel(network_clf, device=args.device, bounds=(0., 1.), preprocessing=foolbox_preprocess(args))
    if args.dataset == 'MNIST':
        rel_stepsize = 0.01/0.3
    else:
        rel_stepsize = 0.25
    if args.att_lp_norm==-1:
        attack = foolbox.attacks.LinfPGD(rel_stepsize=rel_stepsize, steps=args.att_step) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=args.att_eps)
        acc = 1 - success.float().mean(axis=-1)
    elif args.att_lp_norm==2:
        attack = foolbox.attacks.L2PGD(rel_stepsize=rel_stepsize, steps=args.att_step) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=args.att_eps)
        acc = 1 - success.float().mean(axis=-1)
    return x_adv, success, acc