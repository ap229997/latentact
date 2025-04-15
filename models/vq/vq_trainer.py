import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae
from utils.utils import print_current_loss
from utils.metrics import compute_mpjpe, compute_mpjpe_ra, compute_mpjpe_pa, compute_mpjpe_pa_first, binary_classification_metrics
import common.rotation_conversions as rot
from common.quaternion import batch_determinant

import os
import sys
import pickle

def def_value():
    return 0.0


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss(reduction='none')
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss(reduction='none')

            if args.contact_map:
                self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(5.0))

    def forward(self, batch_data, mode='train'):

        if isinstance(batch_data, dict):
            motions = batch_data['motion'].detach().to(self.device).float()
        else:
            motions = batch_data.detach().to(self.device).float()

        if self.opt.pred_cam:
            cam_rot = motions[..., -9:-3].detach().to(self.device).float()
            cam_rot_nondiff = cam_rot
            cam_transl = motions[..., -3:].detach().to(self.device).float()
            motions = motions[..., :-9]
        
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

        known_cam_mask = batch_data['cam_mask'].detach().to(self.device).float().reshape(-1,)
        self.known_cam_mask = known_cam_mask

        bz, ts = motions.shape[:2]
        trainable_mask = torch.ones((bz, ts, self.opt.dim_pose)).to(self.device)
        if self.opt.pred_cam:
            cam_transf = torch.cat([cam_rot_nondiff, cam_transl], dim=-1)
            
            bz, ts = cam_transf.shape[:2]
            cam_transf = cam_transf.reshape(bz*ts, -1)
            
            # replace unknown cam_transf with learnable parameters
            cam_transf[known_cam_mask == 0] = self.vq_model.unknown_cam_transf

            # append cam_transf to motions as last 9 values
            cam_transf = cam_transf.reshape(bz, ts, -1)
            
            motions = torch.cat([motions, cam_transf], dim=-1)

        trainable_mask = trainable_mask * known_cam_mask.reshape(bz, ts, -1)
        
        if self.opt.contact_map:
            gt_contact_map = batch_data['contact_map'].detach().to(self.device).float()
            motions = torch.cat([motions, gt_contact_map], dim=-1)

        if self.opt.decoder_only:
            more_dict['code_idx'] = batch_data['code_idx'].detach().to(self.device).long()
        
        pred_motion, loss_commit, perplexity, more_outs = self.vq_model(motions, **more_dict)
        more_outs['pred_motion'] = pred_motion.clone() # return pred_motion for visualizations and evaluation

        if mode == 'test': # no need to compute losses and metrics when running in test mode
            return more_outs

        more_loss = {}
        if self.opt.contact_map:
            pred_contact_map = pred_motion[..., -778:]
            loss_contact_map = self.cross_entropy(pred_contact_map, gt_contact_map)
            loss_contact_map = torch.mean(loss_contact_map * known_mask.reshape(bz, ts, -1))
            more_loss['contact_map'] = loss_contact_map

            # compute classification metrics
            logits = pred_contact_map.detach().clone()
            labels = gt_contact_map.detach().clone()
            metric_mask = known_mask.reshape(bz, ts, 1)
            precision, recall, f1 = binary_classification_metrics(logits, labels, mask=metric_mask)
            more_loss['precision'] = precision
            more_loss['recall'] = recall
            more_loss['f1'] = f1
            more_loss['logits'] = logits
            more_loss['labels'] = labels

            motions = motions[..., :-778]
            pred_motion = pred_motion[..., :-778]

        loss_rec = self.l1_criterion(pred_motion, motions)
        loss_rec = torch.mean(loss_rec * trainable_mask)

        if self.opt.dataset_name == 'holo' or self.opt.dataset_name == 'arctic':
            loss_explicit = self.l1_criterion(pred_motion, motions)
        else:
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            loss_explicit = self.l1_criterion(pred_local_pos, local_pos)
        loss_explicit = torch.mean(loss_explicit[..., :61]) + torch.mean(loss_explicit[..., -9:])

        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit
        if self.opt.contact_map:
            loss = loss + 0.1*loss_contact_map # hardcoded weight

        if 'joints' in self.opt.motion_type:
            if self.opt.pred_cam:
                self.motions = motions[...,:-9]
                self.pred_motion = pred_motion[...,:-9]
                self.gt_cam_rot = motions[...,-9:-3]
                self.gt_cam_transl = motions[...,-3:]
                self.pred_cam_rot = pred_motion[...,-9:-3]
                self.pred_cam_transl = pred_motion[...,-3:]
            else:
                self.motions = motions
                self.pred_motion = pred_motion
        elif 'mano' in self.opt.motion_type:
            bz, T, _ = motions.shape
            thetas_ = motions[..., :48].reshape(-1, 48)
            betas_ = motions[..., 48:58].reshape(-1, 10)
            joints_mano_ = self.vq_model.mano(global_orient=thetas_[:, :3], hand_pose=thetas_[:, 3:], betas=betas_)
            joints_mano = joints_mano_.joints.reshape(bz, T, -1, 3)
            joints_mano = joints_mano[:, :, self.vq_model.mano_to_openpose, :]
            gt_cam_t = motions[..., 58:61].unsqueeze(2) * self.mano_std[58:61].reshape(1,1,1,-1) + self.mano_mean[58:61].reshape(1,1,1,-1) # unnormalize
            joints_cam = joints_mano + gt_cam_t
            self.motions = joints_cam
            self.gt_cam_t = gt_cam_t

            # repeat for pred_motion
            pred_thetas_ = pred_motion[...,:48].reshape(-1, 48)
            pred_betas_ = pred_motion[..., 48:58].reshape(-1, 10)
            pred_joints_mano_ = self.vq_model.mano(global_orient=pred_thetas_[:, :3], hand_pose=pred_thetas_[:, 3:], betas=pred_betas_)
            pred_joints_mano = pred_joints_mano_.joints.reshape(bz, T, -1, 3)
            pred_joints_mano = pred_joints_mano[:, :, self.vq_model.mano_to_openpose, :]
            pred_cam_t = pred_motion[..., 58:61].unsqueeze(2) * self.mano_std[58:61].reshape(1,1,1,-1) + self.mano_mean[58:61].reshape(1,1,1,-1) # unnormalize
            pred_joints_cam = pred_joints_mano + pred_cam_t
            self.pred_motion = pred_joints_cam
            self.pred_cam_t = pred_cam_t

            gt_joints_ref = batch_data['joints_ref'].detach().to(self.device).float()
            gt_joints_cam_ref = gt_joints_ref.reshape(bz, T, self.opt.joints_num, 3)
            if self.opt.pred_cam:
                pred_rot_cam_ref = rot.rotation_6d_to_matrix(pred_motion[..., -9:-3])
                pred_transl_cam_ref = pred_motion[..., -3:]
                pred_joints_cam_ref = torch.matmul(pred_rot_cam_ref.reshape(-1, 3, 3), pred_joints_cam.reshape(-1, self.opt.joints_num, 3).transpose(1,2)).transpose(1,2) + pred_transl_cam_ref.reshape(-1, 1, 3)
            else:
                gt_rot_cam_ref = batch_data['cam_rot_ref'].detach().to(self.device).float()
                gt_rot_cam_ref = rot.rotation_6d_to_matrix(gt_rot_cam_ref)
                gt_transl_cam_ref = batch_data['cam_transl_ref'].detach().to(self.device).float()
                pred_joints_cam_ref = torch.matmul(pred_joints_cam.reshape(-1, self.opt.joints_num, 3), gt_rot_cam_ref.reshape(-1, 3, 3).transpose(1,2)) + gt_transl_cam_ref.reshape(-1, 1, 3)
            
            pred_joints_cam_ref = pred_joints_cam_ref.reshape(bz, T, self.opt.joints_num, 3)
            self.gt_joints_frame_ref = gt_joints_cam_ref
            self.pred_joints_frame_ref = pred_joints_cam_ref
            joints_ref_mask = batch_data['joints_ref_mask'].detach().to(self.device).float().reshape(-1,)
            combined_mask = joints_ref_mask * self.known_cam_mask
            self.frame_ref_mask = combined_mask

            if self.opt.pred_cam:
                self.gt_cam_rot = motions[..., -9:-3]
                self.gt_cam_transl = motions[..., -3:]
                self.pred_cam_rot = pred_motion[..., -9:-3]
                self.pred_cam_transl = pred_motion[..., -3:]
        else:
            raise KeyError('Motion Type not recognized')

        more_loss.update(more_outs)
        return loss, loss_rec, loss_explicit, loss_commit, perplexity, more_loss

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        
        min_val_loss = np.inf
        logs = defaultdict(def_value, OrderedDict())
            
        if 'mano' in self.opt.motion_type:
            self.mano_mean = torch.from_numpy(train_loader.dataset.mean).to(self.device).float()
            self.mano_std = torch.from_numpy(train_loader.dataset.std).to(self.device).float()

        for epoch in tqdm(range(epoch, self.opt.max_epoch)):
            self.vq_model.train()
            for i, batch_data in enumerate(tqdm(train_loader)):
                it += 1
                
                loss, loss_rec, loss_vel, loss_commit, perplexity, more_loss = self.forward(batch_data)

                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    if self.opt.contact_map:
                        loss_contact_map = more_loss['contact_map']
                        logs['loss_contact_map'] += loss_contact_map.item()

                        logs['contact_precision'] += more_loss['precision']
                        logs['contact_recall'] += more_loss['recall']
                        logs['contact_f1'] += more_loss['f1']

                    curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                               self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                               valid = self.known_cam_mask.reshape(-1))
                    logs['mpjpe'] += curr_mpjpe.item()
                    curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                     self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                     valid = self.known_cam_mask.reshape(-1))
                    logs['mpjpe_ra'] += curr_mpjpe_ra.item()
                    curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                     self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                     valid = self.known_cam_mask.reshape(-1))
                    logs['mpjpe_pa'] += curr_mpjpe_pa.item()
                    curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                                                   self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                                                   valid = self.frame_ref_mask.reshape(-1))
                    logs['mpjpe_ref'] += curr_mpjpe_ref.item()
                    l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
                    l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                    logs['l1_cam_t'] += l1_cam_t.item()

                    bz = self.pred_joints_frame_ref.shape[0]
                    # procrustes alignment at the global level
                    curr_mpjpe_pa_g = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            valid = self.frame_ref_mask.reshape(bz, -1))
                    logs['mpjpe_pa_g'] += curr_mpjpe_pa_g.item()
                    
                    # procrustes alignment at the first frame
                    curr_mpjpe_pa_f = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.window_size, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.window_size, -1, 3),
                                                            valid = self.frame_ref_mask)
                    logs['mpjpe_pa_f'] += curr_mpjpe_pa_f.item()

                    if self.opt.pred_cam:
                        # compute l1 error for cam_rot and cam_transl
                        l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                        l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                        logs['l1_cam_rot_ref'] += l1_cam_rot.item()
                        
                        l1_cam_transl = torch.linalg.norm(self.pred_cam_transl - self.gt_cam_transl, dim=-1)
                        l1_cam_transl = torch.nanmean(l1_cam_transl.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                        logs['l1_cam_transl_ref'] += l1_cam_transl.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    
                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

                if self.opt.debug:
                    break

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            val_epoch = epoch if self.opt.debug else epoch + 1
            if val_epoch % self.opt.val_every_e == 0:
                print('Validation time:')
                self.vq_model.eval()
                val_loss_rec = []
                val_loss_vel = []
                val_loss_commit = []
                val_loss = []
                val_perpexity = []

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    val_mpjpe = []
                    val_mpjpe_ra = []
                    val_mpjpe_pa = []
                    val_mpjpe_pa_f = []
                    val_mpjpe_pa_g = []

                    val_mpjpe_ref = []
                    val_l1_cam_t = []

                    if self.opt.pred_cam:
                        val_l1_cam_rot_ref = []
                        val_l1_cam_transl_ref = []
                    
                    if self.opt.contact_map:
                        val_loss_contact_map = []
                        val_contact_logits = []
                        val_contact_labels = []
                        val_contact_masks = []
                
                indices_out = {}
                with torch.no_grad():
                    for ival, batch_data in enumerate(tqdm(val_loader)):
                        loss, loss_rec, loss_vel, loss_commit, perplexity, more_loss = self.forward(batch_data, mode='val')

                        # move to cpu and detach
                        loss = loss.cpu().detach()
                        loss_rec = loss_rec.cpu().detach()
                        loss_vel = loss_vel.cpu().detach()
                        loss_commit = loss_commit.cpu().detach()
                        perplexity = perplexity.cpu().detach()
                        for k, v in more_loss.items():
                            if isinstance(v, torch.Tensor):
                                more_loss[k] = v.cpu().detach()

                        val_loss.append(loss.item())
                        val_loss_rec.append(loss_rec.item())
                        val_loss_vel.append(loss_vel.item())
                        val_loss_commit.append(loss_commit.item())
                        val_perpexity.append(perplexity.item())
                        
                        if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                            if self.opt.contact_map:
                                loss_contact_map = more_loss['contact_map']
                                val_loss_contact_map.append(loss_contact_map.item())

                                val_contact_logits.append(more_loss['logits'])
                                val_contact_labels.append(more_loss['labels'])
                                val_contact_masks.append(self.known_mask.reshape(-1, self.opt.window_size, 1).cpu())
                            
                            curr_mpjpe = compute_mpjpe(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                               self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                               valid = self.known_cam_mask.reshape(-1))
                            val_mpjpe.append(curr_mpjpe.item())
                            curr_mpjpe_ra = compute_mpjpe_ra(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                            valid = self.known_cam_mask.reshape(-1))
                            val_mpjpe_ra.append(curr_mpjpe_ra.item())
                            curr_mpjpe_pa = compute_mpjpe_pa(self.pred_motion.detach().reshape(-1, self.opt.joints_num, 3), 
                                                            self.motions.detach().reshape(-1, self.opt.joints_num, 3),
                                                            valid = self.known_cam_mask.reshape(-1))
                            val_mpjpe_pa.append(curr_mpjpe_pa)
                            curr_mpjpe_ref = compute_mpjpe(self.pred_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3), 
                                                   self.gt_joints_frame_ref.detach().reshape(-1, self.opt.joints_num, 3),
                                                   valid = self.frame_ref_mask.reshape(-1))
                            val_mpjpe_ref.append(curr_mpjpe_ref.item())
                            
                            l1_cam_t = torch.linalg.norm(self.pred_cam_t - self.gt_cam_t, dim=-1)
                            l1_cam_t = torch.nanmean(l1_cam_t.reshape(-1, 1) * self.known_cam_mask.reshape(-1, 1))
                            val_l1_cam_t.append(l1_cam_t.item())

                            bz = self.pred_joints_frame_ref.shape[0]
                            # procrustes alignment at the global level
                            curr_mpjpe_pa_g = compute_mpjpe_pa(self.pred_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, -1, 3),
                                                            valid = self.frame_ref_mask.reshape(bz, -1))
                            val_mpjpe_pa_g.append(curr_mpjpe_pa_g.item())

                            # procrustes alignment only at the first frame
                            curr_mpjpe_pa_f = compute_mpjpe_pa_first(self.pred_joints_frame_ref.detach().reshape(bz, self.opt.window_size, -1, 3),
                                                            self.gt_joints_frame_ref.detach().reshape(bz, self.opt.window_size, -1, 3),
                                                            valid = self.frame_ref_mask)
                            val_mpjpe_pa_f.append(curr_mpjpe_pa_f.item())

                            if self.opt.pred_cam:
                                # compute l1 error for cam_rot and cam_transl
                                l1_cam_rot = torch.abs(self.pred_cam_rot - self.gt_cam_rot)
                                l1_cam_rot = torch.nanmean(l1_cam_rot.reshape(-1, 6) * self.known_cam_mask.reshape(-1, 1))
                                val_l1_cam_rot_ref.append(l1_cam_rot.item())

                            if self.opt.return_indices:
                                names = batch_data['name']
                                ranges = batch_data['range']
                                indices = more_loss['code_idx']
                                starts = batch_data['start']
                                ends = batch_data['end']
                                bz = indices.shape[0]
                                for b in range(bz):
                                    curr_dict = {'range': (ranges[0][b], ranges[1][b]), 'indices': indices[b].reshape(-1).detach().cpu().numpy()}
                                    if names[b] not in indices_out:
                                        indices_out[names[b]] = {}
                                    indices_out[names[b]][f'{starts[b]}_{ends[b]}'] = curr_dict

                        if self.opt.debug and ival > 0:
                            break

                if self.opt.return_indices:
                    file_name = pjoin(self.opt.model_dir, 'indices_E%04d.pkl' % (val_epoch))
                    with open(file_name, 'wb') as f:
                        pickle.dump(indices_out, f)

                self.logger.add_scalar('Val/loss', np.nanmean(val_loss), epoch)
                self.logger.add_scalar('Val/loss_rec', np.nanmean(val_loss_rec), epoch)
                self.logger.add_scalar('Val/loss_vel', np.nanmean(val_loss_vel), epoch)
                self.logger.add_scalar('Val/loss_commit', np.nanmean(val_loss_commit), epoch)
                self.logger.add_scalar('Val/loss_perplexity', np.nanmean(val_perpexity), epoch)

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    self.logger.add_scalar('Val/mpjpe', np.nanmean(val_mpjpe), epoch)
                    self.logger.add_scalar('Val/mpjpe_ra', np.nanmean(val_mpjpe_ra), epoch)
                    self.logger.add_scalar('Val/mpjpe_pa', np.nanmean(val_mpjpe_pa), epoch)
                    self.logger.add_scalar('Val/mpjpe_pa_f', np.nanmean(val_mpjpe_pa_f), epoch)
                    self.logger.add_scalar('Val/mpjpe_pa_g', np.nanmean(val_mpjpe_pa_g), epoch)

                    self.logger.add_scalar('Val/mpjpe_ref', np.nanmean(val_mpjpe_ref), epoch)
                    self.logger.add_scalar('Val/l1_cam_t', np.nanmean(val_l1_cam_t), epoch)

                    if self.opt.pred_cam:
                        self.logger.add_scalar('Val/l1_cam_rot_ref', np.nanmean(val_l1_cam_rot_ref), epoch)
                        self.logger.add_scalar('Val/l1_cam_transl_ref', np.nanmean(val_l1_cam_transl_ref), epoch)

                    if self.opt.contact_map:
                        self.logger.add_scalar('Val/loss_contact_map', np.nanmean(val_loss_contact_map), epoch)

                        all_logits = torch.cat(val_contact_logits, dim=0)
                        all_labels = torch.cat(val_contact_labels, dim=0)
                        all_masks = torch.cat(val_contact_masks, dim=0)
                        precision, recall, f1 = binary_classification_metrics(all_logits, all_labels, all_masks)
                        self.logger.add_scalar('Val/contact_precision', precision, epoch)
                        self.logger.add_scalar('Val/contact_recall', recall, epoch)
                        self.logger.add_scalar('Val/contact_f1', f1, epoch)

                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f, MPJPE: %.5f' %
                        (np.nanmean(val_loss), np.nanmean(val_loss_rec), 
                        np.nanmean(val_loss_vel), np.nanmean(val_loss_commit), np.nanmean(val_mpjpe_ref)))
                else:
                    print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
                        (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                        sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss)))
                
                curr_val_loss = np.nanmean(val_loss)
                if curr_val_loss is not np.nan and curr_val_loss < min_val_loss:
                    min_val_loss = curr_val_loss
                    min_val_epoch = epoch
                    self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                    print('Best Validation Model So Far!~')
    
    def inference(self, val_loader, ckpt='latest'):
        self.vq_model.to(self.device)

        model_dir = pjoin(self.opt.model_dir, f'{ckpt}.tar')
        if self.opt.transfer_from is not None:
            splits = self.opt.transfer_from.split('/')
            transfer_dataset, transfer_model = splits[-2], splits[-1]
            model_dir = model_dir.replace(f'/{self.opt.dataset_name}/', f'/{transfer_dataset}/')
            model_dir = model_dir.replace(f'/{self.opt.name}/', f'/{transfer_model}/')
        assert os.path.exists(model_dir), f'Model {model_dir} does not exist!'
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        print (f'Loaded model from {model_dir} at epoch {epoch} and iteration {it}')

        if 'mano' in self.opt.motion_type:
            self.mano_mean = torch.from_numpy(val_loader.dataset.mean).to(self.device).float()
            self.mano_std = torch.from_numpy(val_loader.dataset.std).to(self.device).float()

        print('Running inference:')
        self.vq_model.eval()

        indices_out = {}
        with torch.no_grad():
            for ival, batch_data in enumerate(tqdm(val_loader)):
                outs = self.forward(batch_data, mode='test')
                if isinstance(outs, tuple) or isinstance(outs, list):
                    outs = outs[-1]
                
                if 'holo' in self.opt.dataset_name or 'arctic' in self.opt.dataset_name:
                    if self.opt.return_indices:
                        names = batch_data['name']
                        ranges = batch_data['range']
                        indices = outs['code_idx']
                        starts = batch_data['start']
                        ends = batch_data['end']
                        bz = indices.shape[0]
                        for b in range(bz):
                            curr_dict = {'range': (ranges[0][b], ranges[1][b]), 'indices': indices[b].reshape(-1).detach().cpu().numpy()}
                            if names[b] not in indices_out:
                                indices_out[names[b]] = {}
                            indices_out[names[b]][f'{starts[b]}_{ends[b]}'] = curr_dict

                if self.opt.debug and ival > 0:
                    break

        if self.opt.return_indices:
            prefix = ''
            sett = '' if self.opt.setting is None else self.opt.setting
            if self.opt.transfer_from is not None:
                prefix = f'transfer_{transfer_dataset}_{transfer_model}_'
            file_name = pjoin(self.opt.model_dir, f'{prefix}indices_finest_{sett}_{val_loader.dataset.split}.pkl')
            with open(file_name, 'wb') as f:
                pickle.dump(indices_out, f)


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            if not os.path.exists(model_dir):
                print ('No model found, training from scratch')
            else:
                epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()


            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss
