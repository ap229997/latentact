import os
from os.path import join as pjoin
import json
import argparse

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.fixseed import fixseed
from options.vq_option import arg_parse

from models.prior.learned_prior import LearnedPrior
from models.vq.model import RVQVAE
from data.t2m_dataset import get_dataset
from utils.get_opt import get_opt
from utils.metrics import *
from utils.eval_fn import prior_forward_pass, EvalWrapper


if __name__ == '__main__':
    opt = arg_parse(True)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

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
        if opt.pred_cam:
            opt.dim_pose += 6 # for cam rotation
            opt.dim_pose += 3 # for cam translation
    else:
        raise ValueError('Unknown motion type')

    dataset_class = get_dataset(opt.dataset_name)
    dataset = dataset_class(opt, split='test')
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # load prior model
    prior_opt = opt.__dict__.copy()
    prior_opt = argparse.Namespace(**prior_opt)
    prior_opt.dropout = 0.0
    prior_opt.use_contact_points = True
    prior = LearnedPrior(prior_opt).to(opt.device)
    prior_name = opt.load_prior
    if prior_name is None:
        prior_name = opt.name
    prior_path = pjoin(opt.checkpoints_dir, 'prior', prior_name)
    prior_ckpt = pjoin(prior_path, 'model', 'latest.tar')
    prior.load_state_dict(torch.load(prior_ckpt, map_location=opt.device))
    prior.eval()

    # this should already be taken care of internally by EvalWrapper below
    decoder_path = pjoin(opt.checkpoints_dir, opt.eval_model)
    decoder_ckpt = pjoin(decoder_path, 'model', 'latest.tar')
    decoder_opt_path = pjoin(decoder_path, 'opt.txt')
    decoder_opt = get_opt(decoder_opt_path, **opt.__dict__)
    decoder_opt.is_train = False
    net = RVQVAE(decoder_opt,
                decoder_opt.dim_pose,
                decoder_opt.nb_code,
                decoder_opt.code_dim,
                decoder_opt.code_dim,
                decoder_opt.down_t,
                decoder_opt.stride_t,
                decoder_opt.width,
                decoder_opt.depth,
                decoder_opt.dilation_growth_rate,
                decoder_opt.vq_act,
                decoder_opt.vq_norm,
                opt=decoder_opt).to(opt.device)

    eval_wrapper = EvalWrapper(decoder_opt, vq_model=net)
    eval_wrapper.vq_model.load_vqvae_model(decoder_path, decoder=True, quantizer=True, mode='latest', feedforward=opt.feedforward)
    eval_wrapper.vq_model.eval()
    if 'mano' in decoder_opt.motion_type:
        eval_wrapper.mano_mean = torch.from_numpy(loader.dataset.mean).to(decoder_opt.device).float()
        eval_wrapper.mano_std = torch.from_numpy(loader.dataset.std).to(decoder_opt.device).float()

    running_metrics = {}
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader)):
            codebook_indices = prior_forward_pass(prior, data, prior_opt, stochastic=opt.stochastic)
            # update 'code_idx' in data with 'codebook_indices'
            data['code_idx'] = codebook_indices.reshape(-1, decoder_opt.window_size*decoder_opt.num_quantizers)
                
            outs = eval_wrapper.forward(data, mode='val') # used 'test' mode only when losses & metrics are not needed

            curr_metrics = eval_wrapper.compute_metrics(outs)
            if len(running_metrics) == 0:
                # intialize running metrics with empty lists
                for k, v in curr_metrics.items():
                    running_metrics[k] = [v]
            else:
                for k, v in curr_metrics.items():
                    running_metrics[k].append(v)

            if opt.debug and idx == 10:
                break

    all_contact_logits = running_metrics.pop('contact_logits')
    all_contact_labels = running_metrics.pop('contact_labels')
    all_contact_masks = running_metrics.pop('contact_masks')
    all_logits = torch.cat(all_contact_logits, dim=0)
    all_labels = torch.cat(all_contact_labels, dim=0)
    all_masks = torch.cat(all_contact_masks, dim=0)
    precision, recall, f1 = binary_classification_metrics(all_logits, all_labels, all_masks)

    final_metrics = {}
    for k, v in running_metrics.items():
        final_metrics[k] = np.nanmean(v)
    final_metrics['precision'] = precision
    final_metrics['recall'] = recall
    final_metrics['f1'] = f1

    for k, v in final_metrics.items():
        print (f'{k}: {v:.4f}')

    metrics_path = pjoin(decoder_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
      json.dump(final_metrics, f, indent=4)        