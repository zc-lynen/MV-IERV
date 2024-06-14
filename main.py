from __future__ import print_function

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from dataset import *
from model import *
from utils import *



def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--dataset', type=str, default='./data/g', help='dataset directory')
    parser.add_argument('--dataset_basrec', type=str, default='./data/g_rec23', help='basic view reconstruction dataset directory')

    # architecture parameters
    parser.add_argument('--fea_hw_dim', type=str, default='9_16_40', help='feature size (h,w,c) for conv')
    parser.add_argument('--expansion', type=int, default=1, help='channel expansion from grid to conv')
    parser.add_argument('--reduction', type=int, default=2, help='channel reduction of conv layers')
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--norm', default='in', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use',
                        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower_width', type=int, default=96, help='lowest channel width for output feature maps')

    # inter-view prediction
    parser.add_argument('--view_num', type=int, default=10, help='view number of multiview video')
    parser.add_argument('--frame_perview', type=int, default=10, help='frame number in each view')
    parser.add_argument('--basic_ind', type=int, default=0, help='basic view index')
    parser.add_argument('--resol', nargs='+', default=[1920, 1080], type=int, help='frame resolution')

    # general training setups
    parser.add_argument('--gpu_idx', type=int, default=0, help='used gpu index')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5')
    parser.add_argument('--lw', type=float, default=0.1, help='loss weight')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')

    # compression paramaters
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')
    parser.add_argument('--qbit', type=int, default=-1, help='quantization bit')

    # logging, output directory
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='output_dir', help='folder to output images and model checkpoints')

    args = parser.parse_args()
    print(args)

    if args.frame_perview < 2:
        print('frame_perview must >=2')
        sys.exit(0)
    if args.frame_perview % args.batchSize != 0:
        print('Adjust batchSize to be evenly divided by frame_perview')
        sys.exit(0)

    global bas_ind
    bas_ind = args.basic_ind

    args.warmup = int(args.warmup * args.epochs)

    exp_id = f'{args.dataset}/e{args.epochs}'
    if args.eval_only:
        exp_id += '_eval'
    elif args.prune_ratio < 1:
        exp_id += '_prune'
    else:
        exp_id += '_train'
    exp_id += f'_{args.fea_hw_dim}'

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        shutil.rmtree(args.outf)
        print('Will overwrite the existing output dir!')

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    torch.set_printoptions(precision=2)
    torch.cuda.set_device(args.gpu_idx)
    print("Use GPU:{} for training".format(args.gpu_idx))
    train(args.gpu_idx, args)


# initialize basic view from explicit reconstruction
def fill_buffer(ref_data, t_idx, v_idx, local_rank):
    v = v_idx[0]
    if v == bas_ind:
        ref_data = ref_data.cuda(local_rank, non_blocking=True)
        cur_data = ref_data.detach().clone().half()
        basic_idx = t_idx
        frame_buffer[basic_idx.long()] = cur_data


def interview_pred(model, t_idx, v_idx, local_rank, train):
    # model iteration
    t_idx = t_idx.cuda(local_rank, non_blocking=True)
    v_idx = v_idx.cuda(local_rank, non_blocking=True)

    if train:
        output_list = model(t_idx, v_idx)
    else:
        with torch.no_grad():
            output_list = model(t_idx, v_idx)

    flows    = output_list[0][:,:2,:,:]
    wei1     = output_list[0][:,2:,:,:]
    init_rec = output_list[1][:,:3,:,:]
    wei2     = output_list[1][:,3:,:,:]

    if 0:
        return init_rec, init_rec, init_rec
    else:
        # start flow-guided inter-view prediction and compensation
        soft = nn.Softmax(dim=1)
        # reference (basic) view frame index
        basic_idx = t_idx

        # obtain frame
        basic_frame = frame_buffer[basic_idx.long()].to(torch.float32)
        basic_frame = basic_frame.reshape(-1, 3, basic_frame.shape[-2], basic_frame.shape[-1])

        # warping
        flows = flows.reshape(-1, 2, flows.shape[-2], flows.shape[-1])
        warped_frame = resample(basic_frame, flows).unsqueeze(0)
        warped_frame = warped_frame.reshape(-1, 1, 3, warped_frame.shape[-2], warped_frame.shape[-1])

        # warped frame weighting
        wei1 = soft(wei1)
        pred_frame = torch.sum(warped_frame * wei1.unsqueeze(2), dim=1, keepdim=True)

        # frame aggregation: initial reconstruction and prediction frame
        cat_frames = torch.cat([init_rec.unsqueeze(1), pred_frame], dim=1)
        wei2 = soft(wei2).unsqueeze(2)
        final_frame = torch.sum(cat_frames * wei2, dim=1)

        return init_rec, pred_frame, final_frame


def train(local_rank, args):
    cudnn.benchmark = True

    # psnr, ssim
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    # build model
    model = Generator(
        fea_hw_dim=args.fea_hw_dim, expansion=args.expansion, reduction=args.reduction, lower_width=args.lower_width,
        stride_list=args.strides, norm=args.norm, act=args.act, bias=True, sigmoid=args.sigmoid,
        t_dim=args.frame_perview, v_dim=args.view_num, qbit=args.qbit)

    ##### prune model params and flops #####
    prune_net = args.prune_ratio < 1
    if prune_net:
        param_list = []
        param_to_prune = []
        for k,v in model.named_parameters():
            # print(k)
            if 'weight' in k:
                if 'conv_layers' in k and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.conv_layers[layer_ind].conv)
                    # param_to_prune.append([model.conv_layers[layer_ind].conv,'weight'])

        param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    print_str = str(model) + '\n' + f'Model Params: {total_params}M\n'
    print(print_str)
    write_str_train(args.outf, print_str)

    # optimizer
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # resume from args.weight
    checkpoint = None
    loc = f'cuda:{local_rank}'
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        model, checkpoint = load_model(args.weight, model)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print(f'Model sparsity: {sparsity_num / 1e6 / total_params}')

        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch']
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    dataset = CustomDataSet
    img_transforms = transforms.ToTensor()

    train_data_dir = args.dataset
    val_data_dir = args.dataset
    ref_data_dir = args.dataset_basrec

    train_dataset = dataset(train_data_dir, ref_data_dir, img_transforms, time_stamp=args.frame_perview, view_num=args.view_num)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True, worker_init_fn=worker_init_fn)

    val_dataset = dataset(val_data_dir, ref_data_dir, img_transforms, time_stamp=args.frame_perview, view_num=args.view_num)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False, worker_init_fn=worker_init_fn)
    data_size = len(train_dataset)
    
    # basic frame buffer
    global frame_buffer
    frame_buffer = torch.cuda.FloatTensor(args.frame_perview, 3, args.resol[1], args.resol[0]).fill_(0).half()

    # only evaluation
    if args.eval_only:
        print('Evaluation ...')
        print_str = f'Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparsity_num / 1e6 / total_params}\n'

        val_psnr, val_msssim = evaluate(model, val_dataloader, local_rank, args)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.qbit}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        write_str_eval(args.outf, print_str + '\n')
        return

    # training
    start = datetime.now()
    total_epochs = args.epochs
    updated = False

    for epoch in range(args.start_epoch, total_epochs):
        model.train()
        ##### prune the network if needed #####
        if prune_net and epoch in args.prune_steps:
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparsity_num / 1e6 / total_params}')
        
        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []

        # fill the buffer before training
        if updated == False:
            for i, (data, ref_data, t_idx, view_idx, frame_idx) in enumerate(train_dataloader):
                fill_buffer(ref_data, t_idx, view_idx, local_rank)
                
        # iterate over dataloader
        for i, (data, ref_data, t_idx, view_idx, frame_idx) in enumerate(train_dataloader):
            if 1: # v not in bas_ind
                data = data.cuda(local_rank, non_blocking=True)
                init_rec, pred_frame, final_frame = interview_pred(model, t_idx, view_idx, local_rank, True)
                output_list = [init_rec, pred_frame.squeeze(1), final_frame]

                target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
                loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
                loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
                loss_sum = sum(loss_list)
                lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()

                # compute psnr and msssim
                psnr_list.append(psnr_fn(output_list, target_list))
                msssim_list.append(msssim_fn(output_list, target_list))
                if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                    train_psnr, train_msssim = mean_score(psnr_list, msssim_list)
                    time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e}, PSNR: {}, MSSSIM: {}'.format(
                        time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                        RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                    print(print_str, flush=True)
                    write_str_train(args.outf, print_str + '\n')

        updated = True

        # ADD train_PSNR TO TENSORBOARD
        is_train_best, train_best_psnr, train_best_msssim = compare_score(train_psnr, train_msssim, train_best_psnr, train_best_msssim)
        print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(
            args.resol[1], train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
        print(print_str, flush=True)
        write_str_train(args.outf, print_str + '\n')
        epoch_end_time = datetime.now()
        print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),
        }

        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_dataloader, local_rank, args)
            val_end_time = datetime.now()
   
            # ADD val_PSNR TO TENSORBOARD
            print_str = f'Eval best_PSNR at epoch{epoch+1}:'
            is_val_best, val_best_psnr, val_best_msssim = compare_score(val_psnr, val_msssim, val_best_psnr, val_best_msssim)
            print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(args.resol[1], val_psnr[-1].item(),
                    val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
            print(print_str)
            write_str_train(args.outf, print_str + '\n')
            if is_val_best:
                torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
        if is_train_best:
            torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))


@torch.no_grad()
def evaluate(model, val_dataloader, local_rank, args):
    # Saved weights are not applied quantization -> applying qfn
    if args.qbit != -1 and args.eval_only:
        cur_ckt = model.state_dict()
        quant_weight_list = []
        for k,v in cur_ckt.items():
            weight = torch.tanh(v)
            quant_v = qfn.apply(weight, args.qbit)
            valid_quant_v = quant_v
            quant_weight_list.append(valid_quant_v.flatten())
            # cur_ckt[k] = quant_v
        cat_param = torch.cat(quant_weight_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        from dahuffman import HuffmanCodec
        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)         
        encoding_efficiency = avg_bits / args.qbit

        print_str = f'Entropy encoding efficiency for bit {args.qbit}: {encoding_efficiency}'
        print_bits = f'total_bits: {total_bits}'
        print(print_str)
        write_str_eval(args.outf, print_str + '\n' + print_bits + '\n')
        # model.load_state_dict(cur_ckt)

    psnr_list = []
    msssim_list = []
    if args.dump_images:
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []
    model.eval()
    global frame_buffer

    for i, (data, ref_data, t_idx, view_idx, frame_idx) in enumerate(val_dataloader):
        fill_buffer(ref_data, t_idx, view_idx, local_rank)
    
    for i, (data, ref_data, t_idx, view_idx, frame_idx) in enumerate(val_dataloader):
        v = view_idx[0]
        if 1: # v not in bas_ind
            start_time = datetime.now()
            data = data.cuda(local_rank, non_blocking=True)
            _, _, final_frame = interview_pred(model, t_idx, view_idx, local_rank, False)
            output_list = [final_frame]

            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

            # compute psnr and ms-ssim
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
            psnr_list.append(psnr_fn(output_list, target_list))
            msssim_list.append(msssim_fn(output_list, target_list))
            val_psnr, val_msssim = mean_score(psnr_list, msssim_list)
            if i % args.print_freq == 0:
                fps = (i+1) * args.batchSize / sum(time_list)
                print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                    local_rank, i+1, len(val_dataloader),
                    RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
                print(print_str)
                write_str_train(args.outf, print_str + '\n')

        # dump predictions
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                if v == bas_ind: # v not in bas_ind
                    save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                else:
                    save_image(frame_buffer[t_idx], f'{visual_dir}/pred_{full_ind}.png') # full_ind
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

    model.train()

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()