import importlib

from datasets.data_loader import get_loaders
import torch
import numpy as np
# from networks.resnet18 import  resnet18
from networks.resnet32 import resnet32

import argparse
from networks import allmodels,set_tvmodel_head_var
from loggers.exp_logger import MultiLogger
import  utils
from datasets.dataset_config import  dataset_config
from functools import reduce



def main(args):
    # utils.seed_everything(seed=args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, None,
                                                              args.batch_size, num_workers=4,
                                                              pin_memory=True)

    
    
    from networks.network import LLL_Net

    utils.seed_everything(seed=args.seed)
    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net()
    init_model.head_var = 'fc'

    utils.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=True)

    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger("results", full_exp_name, loggers=['tensorboard'], save_models=False)
    logger.log_args(argparse.Namespace(**args.__dict__))

    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    # from approach.lwf import Appr
    utils.seed_everything(seed=args.seed)
    appr = Appr(net,device,nepochs=args.nepochs,lr=args.lr,logger=logger)

    max_task = len(taskcla) 
    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        appr.train(t, trn_loader[t], val_loader[t])
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))

    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # dataset args
   
    parser.add_argument('--batch_size', default=128, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet18', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    # training args
    parser.add_argument('--approach', default='finetuning', type=str,
                    help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=5, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')

    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')

    parser.add_argument('--momentum', default=0.9, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--wd', default=1e-4, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--num-tasks', default=10, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default=%(default)s)')
    # Args -- Incremental Learning Framework
    args = parser.parse_args()
    main(args)