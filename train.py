from ast import arg
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt


def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--data_path', type=str, default=None, help='override opt.data_path from config')
    parser.add_argument('--load_path', type=str, default=None, help='override opt.load_path (e.g., path to SAMUS checkpoint for AutoSAMUS init)')
    parser.add_argument('--unfreeze_encoder', action='store_true', help='unfreeze SAMUS learnable parts in AutoSAMUS (adapters, CNN branch, upneck) for full AutoSAMUS training')

    args = parser.parse_args()
    opt = get_config(args.task)
    if args.data_path is not None:
        opt.data_path = args.data_path
    if args.load_path is not None:
        opt.load_path = args.load_path

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.to(device)
    # For full AutoSAMUS: unfreeze SAMUS learnable parts (adapters, CNN branch, rel_pos, upneck)
    if args.modelname == 'AutoSAMUS' and args.unfreeze_encoder:
        for n, value in model.image_encoder.named_parameters():
            if ("cnn_embed" in n or "post_pos_embed" in n or "Adapter" in n or
                "2.attn.rel_pos" in n or "5.attn.rel_pos" in n or
                "8.attn.rel_pos" in n or "11.attn.rel_pos" in n or "upneck" in n):
                value.requires_grad = True
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path, map_location="cpu")
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # Build parameter groups: use differential lr when --unfreeze_encoder is set
    # Pre-trained encoder parts get 10x lower lr to avoid corruption from random APG gradients
    if args.modelname == 'AutoSAMUS' and args.unfreeze_encoder:
        encoder_params = []
        new_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'image_encoder' in n:
                encoder_params.append(p)
            else:
                new_params.append(p)
        print(f"Differential LR: {len(encoder_params)} encoder params (lr={args.base_lr*0.1:.6f}), "
              f"{len(new_params)} new params (lr={args.base_lr:.6f})")
        param_groups = [
            {'params': new_params, 'lr': args.base_lr, 'initial_lr': args.base_lr},
            {'params': encoder_params, 'lr': args.base_lr * 0.1, 'initial_lr': args.base_lr * 0.1},
        ]
    else:
        param_groups = [{'params': filter(lambda p: p.requires_grad, model.parameters()),
                         'lr': args.base_lr, 'initial_lr': args.base_lr}]

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        for pg in param_groups:
            pg['lr'] = pg['initial_lr'] / args.warmup_period
        optimizer = torch.optim.AdamW(param_groups, lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(param_groups, lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks) 
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            print(f'\r  batch [{batch_idx+1}/{len(trainloader)}] loss: {train_loss.item():.4f}', end='', flush=True)
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                scale = (iter_num + 1) / args.warmup_period
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group.get('initial_lr', args.base_lr) * scale
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    scale = (1.0 - shift_iter / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group.get('initial_lr', args.base_lr) * scale
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print(f'\repoch [{epoch}/{opt.epochs}], train loss: {train_losses / (batch_idx + 1):.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

if __name__ == '__main__':
    main()