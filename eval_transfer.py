from attacks import *
from utils import *
from defense import *
import torch.nn.functional as F
import tqdm

def classifier_attack_and_purif(args, dataloader, clf, trans_to_clf, score, diffusion):
    accuracy, aaccuracy, paccuracy, saccuracy = 0., 0., 0., 0.
    
    cnt = 0
    for i, (x_val, y_val) in enumerate(tqdm.tqdm(dataloader)):
        cnt += x_val.shape[0]
        x_val = x_val.to(args.device).to(torch.float32)
        y_val = y_val.to(args.device).to(torch.long)
        y_val = y_val.view(-1,)

        perturbed_X, _, _ = eval(args.att_method)(x_val, y_val, None, score, clf, args)

        # purify natural and adversarial samples
        if args.purify_model == 'edm':
            purif_X_re = purify_x_edm_one_shot(perturbed_X, score, args)
            purif_X_no_attack_re = purify_x_edm_one_shot(x_val, score, args)
        elif args.purify_model == 'opt':
            if args.purify_method == "x0":
                purif_X_re = purify_x_opt_x0(perturbed_X, score, args)
                purif_X_no_attack_re = purify_x_opt_x0(x_val, score, args)
            elif args.purify_method == "xt":
                purif_X_re = purify_x_opt_xt(perturbed_X, score, args)
                purif_X_no_attack_re = purify_x_opt_xt(x_val, score, args)
            
        
        with torch.no_grad():
            # calculate standard acc (without purification)    
            logit = clf(trans_to_clf(x_val.clone().detach()))
            pred = logit.max(1, keepdim=True)[1].view(-1,).detach()
            acc = (pred == y_val.clone().detach()).float().sum()
            accuracy += acc.cpu().numpy()

            # calculate robust loss and acc (without purification)
            logit = clf(trans_to_clf(perturbed_X.clone().detach()))
            apred = logit.max(1, keepdim=True)[1].view(-1,).detach()
            aacc = (apred == y_val.clone().detach()).float().sum()
            aaccuracy += aacc.cpu().numpy()
            
            # calculate standard loss and acc (with purification)
            logit = clf(trans_to_clf(purif_X_no_attack_re.clone().detach()))
            spred = logit.max(1, keepdim=True)[1].view(-1,).detach()
            sacc = (spred == y_val.clone().detach()).float().sum()
            saccuracy += sacc.cpu().numpy()

            # calculate robust loss and acc (with purification)
            logit = clf(trans_to_clf(purif_X_re.clone().detach()))
            ppred = logit.max(1, keepdim=True)[1].view(-1,).detach()
            pacc = (ppred == y_val.clone().detach()).float().sum()
            paccuracy += pacc.cpu().numpy()

    return 100*accuracy, 100*aaccuracy, 100*saccuracy, 100*paccuracy, cnt