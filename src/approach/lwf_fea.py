import torch
from copy import deepcopy
from argparse import ArgumentParser
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
import numpy as np
import time


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.1, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=1, T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.apha = 5
        self.beta=1

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        # if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
        #     # if there are no exemplars, previous heads are not modified
        #     params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        # else:
        #     params = self.model.parameters()

        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        #
        # if t==3:
        #     # self.apha+=1
        #     self.beta=10
        # if t==6 or t==9:
        #     # self.apha+=1
        #     self.beta+=10

        lr = self.lr
        best_loss = np.inf
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        scheduler = ReduceLROnPlateau(self.optimizer, factor=1. / self.lr_factor, patience=self.lr_patience)
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            # self.logger.log_scalar(task=t, iter=e + 1, name="loss_cos", value=l1, group="train")
            # self.logger.log_scalar(task=t, iter=e + 1, name="loss_l2", value=l2, group="train")
            # self.logger.log_scalar(task=t, iter=e + 1, name="loss_ce", value=l3, group="train")
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')

            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                print(' *', end='')
            if self.optimizer.param_groups[0]['lr'] < self.lr_min:
                print()
                break
            scheduler.step(valid_loss)

            # self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            # self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=self.optimizer.param_groups[0]['lr'], group="train")
            print()
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=self.optimizer.param_groups[0]['lr'],
                                   group="train")
        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:

            targets_old,old_fea = None,None
            if t > 0:
                targets_old,old_fea = self.model_old(images.to(self.device),True)

            outputs,fea = self.model(images.to(self.device),True)
            # loss2,loss1 = 0,0
            # if t>0:
            #     loss1 = self.apha * F.cosine_embedding_loss(fea,old_fea,torch.tensor([1],device=self.device))
            #     tempOuts = self.model_old.headClacify(fea)
            #     loss2 = self.beta*self.fake_criterion(t,tempOuts,targets_old)
            # loss3 = F.cross_entropy(outputs[t],targets.to(self.device)-self.model.task_offset[t])
            # self.fake_criterion(t,outputs,targets,fea,old_fea)
            loss = self.fake_criterion(t,outputs,targets,targets_old,fea,old_fea)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fake_criterion(self,t,outputs,targets,targets_old,fea,old_fea):
        loss1,loss2,loss3 = 0,0,0
        tempOuts=None
        if t>0:
            loss1 = self.apha * F.cosine_embedding_loss(fea, old_fea, torch.tensor([1], device=self.device))
            tempOuts = self.model_old.headClacify(fea)
        for t_old in range(0, t):
            loss2 += F.mse_loss(tempOuts[t_old],targets_old[t_old])

        loss3 = F.cross_entropy(outputs[t], targets.to(self.device) - self.model.task_offset[t])
        return loss1+loss2+loss3

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # images,targets = images.to(self.device),targets.to(self.device)
                # Forward old model
                outputs,fea = self.model(images.to(self.device),True)
                targets_old,old_fea = None,None
                if t > 0:
                    targets_old,old_fea = self.model_old(images.to(self.device),True)
                # # Forward current model
                # loss2, loss1 = 0, 0
                # if t > 0:
                #     loss1 = F.cosine_embedding_loss(fea, old_fea, torch.tensor([1], device=self.device))
                #     tempOuts = self.model_old.headClacify(fea)
                #     loss2 = self.fake_criterion(t, tempOuts, targets_old)
                # loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                # loss3 = F.cross_entropy(outputs[t],targets.to(self.device)-self.model.task_offset[t])
                # loss = 3*loss1+3*loss2+loss3
                # loss = loss3
                loss = self.fake_criterion(t,outputs,targets,targets_old,fea,old_fea)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num



    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
