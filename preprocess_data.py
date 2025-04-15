import torch
from options.vq_option import arg_parse
from utils.fixseed import fixseed
from data.t2m_dataset import get_dataset


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)
    fixseed(opt.seed)

    torch.set_num_threads(opt.num_threads)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")
    
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

    dataset_class = get_dataset(opt.dataset_name)
    train_dataset = dataset_class(opt)
    val_dataset = dataset_class(opt, split='val')
    test_dataset = dataset_class(opt, split='test')