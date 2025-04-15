import os
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from data.t2m_dataset import get_dataset
from utils.fixseed import fixseed
from options.vq_option import arg_parse
from models.prior.learned_prior import LearnedPrior


if __name__ == '__main__':
    opt = arg_parse(True)
    fixseed(opt.seed)

    torch.set_num_threads(opt.num_threads)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    opt.save_root = pjoin(opt.checkpoints_dir, 'prior', opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # keep these params consistent with the model
    if 'joints' in opt.motion_type:
        opt.joints_num = 21
        if 'tf' in opt.model_type:
            opt.dim_pose = 3
        else:
            opt.dim_pose = 63
    elif 'mano' in opt.motion_type:
        opt.joints_num = 21
        opt.dim_pose = 48+10+3 # 48 for thetas, 10 for betas, 3 for cam_t
        
        if opt.pred_cam: # are additional params needed or can these can be subsumed into mano params (doesn't work well)
            opt.dim_pose += 6 # for cam rotation
            opt.dim_pose += 3 # for cam translation
    else:
        raise ValueError('Unknown motion type')

    dataset_class = get_dataset(opt.dataset_name, indices=True)
    train_data = dataset_class(opt, split='train')
    val_data = dataset_class(opt, split='val')
    
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    opt.dropout = 0.0
    opt.use_contact_points = True
    model = LearnedPrior(opt).to(opt.device)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    # add tensorboard logging
    writer = SummaryWriter(log_dir=opt.log_dir)

    len_train = len(train_loader)
    best_val_loss = float('inf')
    for epoch in tqdm(range(opt.max_epoch)):
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            text_feats = data['text_feats'].to(opt.device).float()
            video_feats = data['video_feats'].to(opt.device).float()
            contact_map = data['contact'].to(opt.device).float()
            contact_mask = data['contact_mask'].to(opt.device).float()
            grid_map = data['grid'].to(opt.device).float()
            grid_mask = data['grid_mask'].to(opt.device).float()
            contact_point = data['contact_point'].to(opt.device).float()
            indices = data['code_idx'].to(opt.device).long()
            loss_mask = data['rel_mask'].to(opt.device).float()

            optimizer.zero_grad()
            logits = model(text_feats, video_feats, contact_map, contact_mask, grid_map, grid_mask, contact_point)
            loss = criterion(logits.reshape(-1, opt.nb_code), indices.reshape(-1))
            loss_mask = loss_mask.view(-1,1).repeat(1, opt.window_size * opt.num_quantizers)
            loss = torch.mean(loss * loss_mask.reshape(-1))
            loss.backward()
            optimizer.step()

            if i % opt.log_every == 0:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len_train + i)
                # print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item()}")

            if opt.debug:
                break

        if epoch % opt.save_every_e == 0:
            torch.save(model.state_dict(), os.path.join(opt.model_dir, f'latest.tar'))

        val_epoch = epoch if opt.debug else epoch + 1
        if val_epoch % opt.val_every_e == 0:
            
            model.eval()
            with torch.no_grad():
                val_loss = []
                all_logits, all_indices, all_mask = [], [], []
                for i, data in enumerate(val_loader):
                    text_feats = data['text_feats'].to(opt.device).float()
                    video_feats = data['video_feats'].to(opt.device).float()
                    contact_map = data['contact'].to(opt.device).float()
                    contact_mask = data['contact_mask'].to(opt.device).float()
                    grid_map = data['grid'].to(opt.device).float()
                    grid_mask = data['grid_mask'].to(opt.device).float()
                    contact_point = data['contact_point'].to(opt.device).float()
                    indices = data['code_idx'].to(opt.device).long()
                    loss_mask = data['rel_mask'].to(opt.device).float()

                    logits = model(text_feats, video_feats, contact_map, contact_mask, grid_map, grid_mask, contact_point)
                    # these logits can be used with gumbel_sample() for stochastic sampling
                    loss = criterion(logits.reshape(-1, opt.nb_code), indices.reshape(-1))
                    loss_mask = loss_mask.view(-1,1).repeat(1, opt.window_size * opt.num_quantizers)
                    loss = torch.mean(loss * loss_mask.reshape(-1))
                    val_loss.append(loss.item())

                    if opt.debug:
                        break

                val_loss = np.sum(val_loss) / len(val_loss)
                # print(f"Epoch {epoch}, Val Loss: {val_loss}")
                # log the val loss
                writer.add_scalar('Val/Loss', val_loss, epoch)

                # save the model if it has the best val loss so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(opt.model_dir, f'finest.tar'))

    writer.close()
    print ("Done!")