import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    orig_ckt = checkpoint['state_dict']
    new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()}
    if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
        new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
        model.load_state_dict(new_ckt)
    elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
        model.module.load_state_dict(new_ckt)
    else:
        model.load_state_dict(new_ckt)

    return model, checkpoint


def loss_fn(pred, target, args):
    target = target.detach()

    loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))

    return loss


def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr


def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim


def mean_score(psnr_list, msssim_list):
    psnr = torch.cat(psnr_list, dim=0)               #(batchsize, num_stage)
    psnr = torch.mean(psnr, dim=0)             #(num_stage)
    msssim = torch.cat(msssim_list, dim=0)           #(batchsize, num_stage)
    msssim = torch.mean(msssim.float(), dim=0) #(num_stage)

    return psnr, msssim


def compare_score(psnr, msssim, best_psnr, best_msssim):
    is_best = psnr[-1] > best_psnr
    best_psnr = psnr[-1] if psnr[-1] > best_psnr else best_psnr
    best_msssim = msssim[-1] if msssim[-1] > best_msssim else best_msssim

    return is_best, best_psnr, best_msssim


def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str


def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, args):
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)

    if cur_epoch < args.warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def write_str_train(out_dir, print_str):
    with open('{}/rank0.txt'.format(out_dir), 'a') as f:
        f.write(print_str)


def write_str_eval(out_dir, print_str):
    with open('{}/eval.txt'.format(out_dir), 'a') as f:
        f.write(print_str)