import numpy as np
from scipy import linalg

import torch
from models.vq.vq_trainer import RVQTokenizerTrainer
from models.vq.quantizer import gumbel_sample
from utils.metrics import *


def prior_forward_pass(model, data, opt, stochastic=False):
    text_feats = data['text_feats'].to(opt.device).float()
    video_feats = data['video_feats'].to(opt.device).float()
    contact_map = data['contact'].to(opt.device).float()
    contact_mask = data['contact_mask'].to(opt.device).float()
    grid_map = data['grid'].to(opt.device).float()
    grid_mask = data['grid_mask'].to(opt.device).float()
    contact_point = data['contact_point'].to(opt.device).float()

    if len(text_feats.shape) == 3: # multiple timesteps, only take the first
        text_feats = text_feats[:, 0]
        video_feats = video_feats[:, 0]
        contact_map = contact_map[:, 0]
        contact_mask = contact_mask[:, 0]
        contact_point = contact_point[:, 0]

    logits = model(text_feats, video_feats, contact_map, contact_mask, grid_map, grid_mask, contact_point)

    # logits: (B, T*opt.num_quantizers, opt.nb_code)
    if stochastic:
        # sample from logits
        indices = gumbel_sample(logits, opt.sample_codebook_temp, stochastic=True, training=True)
    else:
        # get indices from logits
        _, indices = torch.max(logits, dim=-1)
    indices = indices.reshape(-1, opt.window_size, opt.num_quantizers)
    return indices


def decoder_forward_pass(model, batch_data, opt, code_idx):
    
    more_dict = {}
    if opt.video_feats:
        video_feats = batch_data['video_feats'].detach().to(opt.device).float()
        if len(video_feats.shape) == 2:
            video_feats = video_feats.unsqueeze(1).repeat(1, opt.window_size, 1)
        more_dict['video_feats'] = video_feats
    if opt.text_feats:
        text_feats = batch_data['text_feats'].detach().to(opt.device).float()
        if len(text_feats.shape) == 2:
            text_feats = text_feats.unsqueeze(1).repeat(1, opt.window_size, 1)
        more_dict['text_feats'] = text_feats
    if opt.contact_grid is not None:
        contact = batch_data['contact'].detach().to(opt.device).float()
        if len(contact.shape) == 4:
            contact = contact.unsqueeze(1).repeat(1, opt.window_size, 1, 1, 1)
        more_dict['contact'] = contact
        grid = batch_data['grid'].detach().to(opt.device).float()
        more_dict['grid'] = grid
        contact_mask = batch_data['contact_mask'].detach().to(opt.device).float()
        if len(contact_mask.shape) == 1:
            contact_mask = contact_mask.unsqueeze(1).repeat(1, opt.window_size)
        more_dict['contact_mask'] = contact_mask
        grid_mask = batch_data['grid_mask'].detach().to(opt.device).float()
        more_dict['grid_mask'] = grid_mask

        if opt.coord_sys == 'contact':
            cam_t_contact = batch_data['cam_t_contact'].detach().to(opt.device).float()
            more_dict['cam_t_contact'] = cam_t_contact
            
            cam_t_contact_ref = batch_data['cam_t_contact_ref'].detach().to(opt.device).float()
            more_dict['cam_t_contact_ref'] = cam_t_contact_ref
    
    code_emb = model.quantizer.get_codebook_entry(code_idx)
    x_out = model.decoder(code_emb, **more_dict) # (bs, T, opt.dim_pose)

    return x_out


def model_forward_pass(model, batch_data, opt, mode='val'):
    if isinstance(batch_data, dict):
        motions = batch_data['motion'].detach().to(opt.device).float()
    else:
        motions = batch_data.detach().to(opt.device).float()

    if opt.pred_cam:
        cam_rot = motions[..., -9:-3].detach().to(opt.device).float()
        cam_rot_nondiff = cam_rot
        cam_transl = motions[..., -3:].detach().to(opt.device).float()
        motions = motions[..., :-9]
    
    more_dict = {}
    if opt.video_feats:
        video_feats = batch_data['video_feats'].detach().to(opt.device).float()
        more_dict['video_feats'] = video_feats
    if opt.text_feats:
        text_feats = batch_data['text_feats'].detach().to(opt.device).float()
        more_dict['text_feats'] = text_feats
    if opt.contact_grid is not None:
        contact = batch_data['contact'].detach().to(opt.device).float()
        more_dict['contact'] = contact
        grid = batch_data['grid'].detach().to(opt.device).float()
        more_dict['grid'] = grid
        contact_mask = batch_data['contact_mask'].detach().to(opt.device).float()
        more_dict['contact_mask'] = contact_mask
        grid_mask = batch_data['grid_mask'].detach().to(opt.device).float()
        more_dict['grid_mask'] = grid_mask

    known_cam_mask = batch_data['cam_mask'].detach().to(opt.device).float().reshape(-1,)

    bz, ts = motions.shape[:2]
    trainable_mask = torch.ones((bz, ts, opt.dim_pose)).to(opt.device)
    if opt.pred_cam:
        cam_transf = torch.cat([cam_rot_nondiff, cam_transl], dim=-1)
        
        bz, ts = cam_transf.shape[:2]
        cam_transf = cam_transf.reshape(bz*ts, -1)
        
        # replace unknown cam_transf with learnable parameters
        cam_transf[known_cam_mask == 0] = model.unknown_cam_transf

        # append cam_transf to motions as last 9 values
        cam_transf = cam_transf.reshape(bz, ts, -1)
        
        motions = torch.cat([motions, cam_transf], dim=-1)

    trainable_mask = trainable_mask * known_cam_mask.reshape(bz, ts, -1)
    
    if opt.contact_map:
        gt_contact_map = batch_data['contact_map'].detach().to(opt.device).float()
        motions = torch.cat([motions, gt_contact_map], dim=-1)
    
    pred_motion, loss_commit, perplexity, more_outs = model(motions, **more_dict)

    return pred_motion


def encoder_forward_pass(model, motions, return_quantized=False):
    bz, ts = motions.shape[:2]
    x = model.preprocess(motions)
    x_enc = model.encoder(x)
    if return_quantized:
        code_idx, x_enc = model.quantizer.quantize(x_enc, return_latent=True)
        x_enc = x_enc.sum(dim=0)
    x_enc = model.postprocess(x_enc)
    return x_enc


class EvalWrapper(RVQTokenizerTrainer):
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        # making sure
        if args.recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss(reduction='none')
        elif args.recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss(reduction='none')

        if args.contact_map:
            self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(5.0)) # check if  this is helpful

    def compute_metrics(self, outs):
        if isinstance(outs, tuple) or isinstance(outs, list):
            outs = outs[-1]
        metrics = {}
        if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
            if self.opt.contact_map:
                metrics['contact_logits'] = outs['logits']
                metrics['contact_labels'] = outs['labels']
                metrics['contact_masks'] = self.known_mask.reshape(-1, self.opt.window_size, 1).cpu()
            
            curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                valid = self.known_cam_mask.reshape(-1))
            metrics['mpjpe'] = curr_mpjpe.item()
            curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                            valid = self.known_cam_mask.reshape(-1))
            metrics['mpjpe_ra'] = curr_mpjpe_ra.item()
            curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                            valid = self.known_cam_mask.reshape(-1))
            metrics['mpjpe_pa'] = curr_mpjpe_pa.item()
            curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                                    self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                                    valid = self.frame_ref_mask.reshape(-1))
            metrics['mpjpe_ref'] = curr_mpjpe_ref.item()

            l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
            l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
            metrics['l1_cam_t'] = l1_cam_t.item()

            bz = self.pred_joints_frame_ref.shape[0]
            # procrustes alignment at the global level
            curr_mpjpe_pa_g = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                            self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                            valid = self.frame_ref_mask.reshape(bz, -1))
            metrics['mpjpe_pa_g'] = curr_mpjpe_pa_g.item()

            # procrustes alignment only at the first frame
            curr_mpjpe_pa_f = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.window_size, -1, 3),
                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.window_size, -1, 3),
                                            valid = self.frame_ref_mask)
            metrics['mpjpe_pa_f'] = curr_mpjpe_pa_f.item()

            if self.opt.pred_cam:
                # compute l1 error for cam_rot and cam_transl
                l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                metrics['l1_cam_rot_ref'] = l1_cam_rot.item()
                
                if not self.opt.coord_sys == 'contact':
                    l1_cam_transl = torch.linalg.norm(self.pred_cam_transl - self.gt_cam_transl, dim=-1)
                    l1_cam_transl = torch.nanmean(l1_cam_transl.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                    metrics['l1_cam_transl_ref'] = l1_cam_transl.item()
        return metrics

    def demo(self, batch_data):
        
        more_dict = {}
        if self.opt.video_feats:
            video_feats = batch_data['video_feats'].detach().to(self.device).float()
            more_dict['video_feats'] = video_feats
        if self.opt.text_feats:
            text_feats = batch_data['text_feats'].detach().to(self.device).float()
            more_dict['text_feats'] = text_feats
        if self.opt.contact_grid is not None:
            contact = batch_data['contact'].detach().to(self.device).float()
            more_dict['contact'] = contact
            grid = batch_data['grid'].detach().to(self.device).float()
            more_dict['grid'] = grid
            contact_mask = batch_data['contact_mask'].detach().to(self.device).float()
            more_dict['contact_mask'] = contact_mask
            known_mask = contact_mask.reshape(-1,)
            self.known_mask = known_mask
            grid_mask = batch_data['grid_mask'].detach().to(self.device).float()
            more_dict['grid_mask'] = grid_mask

        if self.opt.decoder_only:
            more_dict['code_idx'] = batch_data['code_idx'].detach().to(self.device).long()
        
        bz = text_feats.shape[0]
        motion = torch.ones((bz, self.opt.window_size, self.opt.dim_pose)).to(self.device) # dummy motion, added here so that code doesn't break
        pred_motion, loss_commit, perplexity, more_outs = self.vq_model(motion, **more_dict)
        more_outs['pred_motion'] = pred_motion.clone()
        return more_outs