import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse
from data.t2m_dataset import get_dataset
from utils.fixseed import fixseed

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    torch.set_num_threads(opt.num_threads)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    
    if 'joints' in opt.motion_type:
        opt.joints_num = 21
        if 'tf' in opt.model_type:
            opt.dim_pose = 3
        else:
            opt.dim_pose = 63
    elif 'mano' in opt.motion_type:
        opt.joints_num = 21
        opt.dim_pose = 48+10+3 # 48 for thetas, 10 for betas, 3 for cam_t
        
        if opt.pred_cam: # are additional params needed or can these can be subsumed into mano params (doesn't work for well)
            opt.dim_pose += 6 # for cam rotation
            opt.dim_pose += 3 # for cam_t
    else:
        raise ValueError('Unknown motion type')
    
    # kinematic_chain = [[0,13,14,15,16], [0,1,2,3,17], [0,4,5,6,18], [0,10,11,12,19], [0,7,8,9,20]] # mano
    kinematic_chain = [[0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16], [0,17,18,19,20]] # openpose

    net = RVQVAE(opt,
                opt.dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())
    print('Total model parameters: {}M'.format(pc_vq/1000_000))

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Trainable parameters: {}M'.format(trainable_params/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    dataset_class = get_dataset(opt.dataset_name)
    train_dataset = dataset_class(opt)
    val_dataset = dataset_class(opt, split='val')

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=False, num_workers=opt.num_workers,
                            shuffle=False, pin_memory=True)
    
    trainer.train(train_loader, val_loader)