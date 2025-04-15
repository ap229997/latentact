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
    opt.is_train = False
    if 'joints' in opt.motion_type:
        opt.joints_num = 21
        if 'tf' in opt.model_type:
            opt.dim_pose = 3
        else:
            opt.dim_pose = 63
    elif 'mano' in opt.motion_type:
        opt.joints_num = 21
        opt.dim_pose = 48+10+3 # 48 for thetas, 10 for betas, 3 for cam_t
        
        if opt.pred_cam:
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
                opt.vq_norm,
                opt=opt)

    pc_vq = sum(param.numel() for param in net.parameters())
    print('Total model parameters: {}M'.format(pc_vq/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    dataset_class = get_dataset(opt.dataset_name)
    train_dataset = dataset_class(opt)
    loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=False, num_workers=opt.num_workers,
                            shuffle=False, pin_memory=True)
    trainer.inference(loader)

    # run the same for val
    val_dataset = dataset_class(opt, split='val')
    loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=False, num_workers=opt.num_workers,
                            shuffle=False, pin_memory=True)
    trainer.inference(loader)