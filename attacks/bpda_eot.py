import torch
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()

class BPDA_EOT_Attack():
    def __init__(self, model, args):
        self.model = model

        self.config = {
            'eot_defense_ave': 'logits',
            'eot_attack_ave': 'logits',
            'eot_defense_reps': args.eot_defense_reps,
            'eot_attack_reps': args.eot_attack_reps,
            'adv_steps': args.att_n_iter,
            'adv_norm': 'l_inf' if args.att_lp_norm == -1 else 'l_2',
            'adv_eps': args.att_eps,
            'adv_eta': args.att_eps / 4.0,
            'log_freq': 10
        }

        self.tag = True if args.att_method == "pgd_eot" else False

        print(f'BPDA_EOT config: {self.config}')
        print(f'BPDA_EOT tag: {self.tag}')

    def purify(self, x):
        return self.model.purify(x)

    def eot_defense_prediction(self, logits, reps=1, eot_defense_ave=None):
        if eot_defense_ave == 'logits':
            logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif eot_defense_ave == 'softmax':
            logits_pred = F.softmax(logits, dim=1).view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif eot_defense_ave == 'logsoftmax':
            logits_pred = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif reps == 1:
            logits_pred = logits
        else:
            raise RuntimeError('Invalid ave_method_pred (use "logits" or "softmax" or "logsoftmax")')
        _, y_pred = torch.max(logits_pred, 1)
        return y_pred

    def eot_attack_loss(self, logits, y, reps=1, eot_attack_ave='loss'):
        if eot_attack_ave == 'logits':
            logits_loss = logits.view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
            y_loss = y
        elif eot_attack_ave == 'softmax':
            logits_loss = torch.log(F.softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0))
            y_loss = y
        elif eot_attack_ave == 'logsoftmax':
            logits_loss = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
            y_loss = y
        elif eot_attack_ave == 'loss':
            logits_loss = logits
            y_loss = y.repeat(reps)
        else:
            raise RuntimeError('Invalid ave_method_eot ("logits", "softmax", "logsoftmax", "loss")')
        loss = criterion(logits_loss, y_loss)
        return loss

    def predict(self, X, y, requires_grad=True, reps=1, eot_defense_ave=None, eot_attack_ave='loss'):
        if requires_grad:
            if self.tag:
                logits = self.model.alternative(X)
            else:
                logits = self.model.classify(X)
        else:
            with torch.no_grad():
                if self.tag:
                    logits = self.model.alternative(X.data)
                else:
                    logits = self.model.classify(X.data)

        if self.tag:
            logits_true = self.model.purify_and_classify(X)
            y_pred = self.eot_defense_prediction(logits_true.detach(), reps, eot_defense_ave)
        else:
            y_pred = self.eot_defense_prediction(logits.detach(), reps, eot_defense_ave)
        correct = torch.eq(y_pred, y)
        loss = self.eot_attack_loss(logits, y, reps, eot_attack_ave)

        return correct.detach(), loss

    def pgd_update(self, X_adv, grad, X, adv_norm, adv_eps, adv_eta, eps=1e-10):
        if adv_norm == 'l_inf':
            X_adv.data += adv_eta * torch.sign(grad)
            X_adv = torch.clamp(torch.min(X + adv_eps, torch.max(X - adv_eps, X_adv)), min=0, max=1)
        elif adv_norm == 'l_2':
            X_adv.data += adv_eta * grad / grad.view(X.shape[0], -1).norm(p=2, dim=1).view(X.shape[0], 1, 1, 1)
            dists = (X_adv - X).view(X.shape[0], -1).norm(dim=1, p=2).view(X.shape[0], 1, 1, 1)
            X_adv = torch.clamp(X + torch.min(dists, adv_eps*torch.ones_like(dists))*(X_adv-X)/(dists+eps), min=0, max=1)
        else:
            raise RuntimeError('Invalid adv_norm ("l_inf" or "l_2"')
        return X_adv

    def purify_and_predict(self, X, y, purify_reps=1, requires_grad=True):
        X_repeat = X.repeat([purify_reps, 1, 1, 1])
        if self.tag:
            X_repeat.requires_grad_()
            correct, loss = self.predict(X_repeat, y, requires_grad, purify_reps,
                                        self.config['eot_defense_ave'], self.config['eot_attack_ave'])
            if requires_grad:
                X_grads = torch.autograd.grad(loss, [X_repeat])[0]
                attack_grad = X_grads.view([purify_reps]+list(X.shape)).mean(dim=0)
                return correct, attack_grad
            else:
                return correct, None
        else:
            X_repeat_purified = self.purify(X_repeat).detach().clone()
            X_repeat_purified.requires_grad_()
            correct, loss = self.predict(X_repeat_purified, y, requires_grad, purify_reps,
                                        self.config['eot_defense_ave'], self.config['eot_attack_ave'])
            if requires_grad:
                X_grads = torch.autograd.grad(loss, [X_repeat_purified])[0]
                attack_grad = X_grads.view([purify_reps]+list(X.shape)).mean(dim=0)
                return correct, attack_grad
            else:
                return correct, None

    def eot_defense_verification(self, X_adv, y, correct, defended):
        for verify_ind in range(correct.nelement()):
            if correct[verify_ind] == 0 and defended[verify_ind] == 1:
                defended[verify_ind] = self.purify_and_predict(X_adv[verify_ind].unsqueeze(0), y[verify_ind].view([1]),
                                                            self.config['eot_defense_reps'], requires_grad=False)[0]
        return defended

    def eval_and_bpda_eot_grad(self, X_adv, y, defended, requires_grad=True):
        correct, attack_grad = self.purify_and_predict(X_adv, y, self.config['eot_attack_reps'], requires_grad)
        if self.config['eot_defense_reps'] > 0:
            defended = self.eot_defense_verification(X_adv, y, correct, defended)
        else:
            defended *= correct
        return defended, attack_grad

    def attack_batch(self, X, y):
        defended = self.eval_and_bpda_eot_grad(X, y, torch.ones_like(y).bool(), False)[0]
        print('Baseline: {} of {}'.format(defended.sum(), len(defended)))

        class_batch = torch.zeros([self.config['adv_steps'] + 2, X.shape[0]]).bool()
        class_batch[0] = defended.cpu()
        ims_adv_batch = torch.zeros(X.shape)
        for ind in range(defended.nelement()):
            if defended[ind] == 0:
                ims_adv_batch[ind] = X[ind].cpu()

        X_adv = X.clone()

        for step in range(self.config['adv_steps'] + 1):
            defended, attack_grad = self.eval_and_bpda_eot_grad(X_adv, y, defended)

            class_batch[step+1] = defended.cpu()
            for ind in range(defended.nelement()):
                if class_batch[step, ind] == 1 and defended[ind] == 0:
                    ims_adv_batch[ind] = X_adv[ind].cpu()

            if step < self.config['adv_steps']:
                X_adv = self.pgd_update(X_adv, attack_grad, X, self.config['adv_norm'], self.config['adv_eps'], self.config['adv_eta'])
                X_adv = X_adv.detach().clone()

            if step == 1 or step % self.config['log_freq'] == 0 or step == self.config['adv_steps']:
                print('Attack {} of {}   Batch defended: {} of {}'.
                    format(step, self.config['adv_steps'], int(torch.sum(defended).cpu().numpy()), X_adv.shape[0]))

            if int(torch.sum(defended).cpu().numpy()) == 0:
                print('Attack successfully to the batch!')
                break

        for ind in range(defended.nelement()):
            if defended[ind] == 1:
                ims_adv_batch[ind] = X_adv[ind].cpu()

        return class_batch, ims_adv_batch