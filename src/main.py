import importlib

from datasets.data_loader import get_loaders
import torch
import numpy as np

import argparse
from networks import allmodels,set_tvmodel_head_var
from loggers.exp_logger import MultiLogger
import  utils
from datasets.dataset_config import  dataset_config
from functools import reduce
import time
import os


def main(args):
    tstart = time.time()
    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

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
    logger = MultiLogger("results", full_exp_name, loggers=['tensorboard','disk'], save_models=True)
    logger.log_args(argparse.Namespace(**args.__dict__))


    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    utils.seed_everything(seed=args.seed)
    appr = Appr(net,device,nepochs=args.nepochs,lr=args.lr,logger=logger,wd=args.wd)


    max_task = len(taskcla) 
    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
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
            test_loss, acc_taw[t, u],_ = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]

            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u]))
        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t)
        logger.log_result(forg_taw, name="forg_taw", step=t)
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)

    utils.print_summary(acc_taw, forg_taw)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

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