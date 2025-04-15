from typing import Any
import os
import sys
sys.path.append(os.path.dirname(__file__))
import pickle
import random
import argparse
import json
import glob

import numpy as np
from PIL import Image, ImageDraw
from glob import glob
from tqdm import tqdm

import torch
import common.rotation_conversions as rot_conv
from torch.utils.data import Dataset

from common.generic_utils import *
from common.holo_utils import get_video_subset, sample_clips
from common.mano_wrapper import MANO, MANO_PATH, LIGHT_BLUE
from common.vis_utils import MeshRenderer


class HoloContacts(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        
        video_names = self.load_clips(args)

        self.motion_data = self.sample_valid_frames(video_names)

        ##### another source of hands masks from hands23 model #####
        # # this is useful when the concave hull fails to create a mask from hamer predictions
        # hand_mask_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_hand_masks')
        # self.hand_mask_dir = hand_mask_dir
        # self.load_hand_masks(hand_mask_dir)

        action_path = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_action_data')
        self.load_action_data(action_path)

        obj_mask_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_obj_masks')
        self.obj_mask_dir = obj_mask_dir
        self.load_obj_masks(obj_mask_dir)

        self.mano_right = MANO(model_path=MANO_PATH, gender='neutral', num_hand_joints=15, create_body_pose=False)
        # instantiate the renderer
        self.renderer = MeshRenderer(self.mano_right.faces)

        self.process_actions()

        self.process_motions()

    def __len__(self) -> int:
        return len(self.motion_data)

    def load_clips(self, args) -> None:
        annot_path = os.path.join(os.path.dirname(os.path.abspath(args.img_folder)), 'data-annotation-trainval-v1_1.json')
        annot = json.load(open(annot_path, 'r'))

        name2idx, idx2name = {}, {}
        for i in range(len(annot)):
            name2idx[annot[i]['video_name']] = i
            idx2name[i] = annot[i]['video_name']

        video_names = get_video_subset(annot, args.task_type) # dict of lists

        if args.subsample is not None:
            clips = sample_clips(video_names, args.subsample) # list of dicts
            # convert back to dict of lists
            video_names = {}
            for clip in clips:
                if clip['video_name'] not in video_names:
                    video_names[clip['video_name']] = []
                video_names[clip['video_name']].append(clip['clip'])
            # sort video names by keys
            video_names = dict(sorted(video_names.items()))

        # extract start_num to end_num video_names
        if args.start_num is None:
            args.start_num = 0
        if args.end_num is None:
            args.end_num = len(video_names)
        video_names = dict(list(video_names.items())[args.start_num:args.end_num])

        return video_names

    def sample_valid_frames(self, video_names):
        hamer_pred_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_hamer_preds')
        all_hamer = os.listdir(hamer_pred_dir)
        all_hamer = [x.replace('_masks_hamer.pkl', '') for x in all_hamer if 'visor' not in x]
        rel_motion_data = [] # list of dicts
        total_frames = 0
        for j, name in enumerate(video_names):
            if name not in all_hamer:
                print (f'No hamer predictions found for {name}.')
                continue
            
            pkl_file = f'{hamer_pred_dir}/{name}_masks_hamer.pkl'
            curr_pkl = pickle.load(open(pkl_file, 'rb'))
            
            for clip in video_names[name]:
                curr_start, curr_end, verb, noun, hand_type = clip
                curr_hand_type = 'right'
                valid_idx = [idx for idx in range(curr_start, curr_end+1) if idx in curr_pkl]
                if len(valid_idx) == 0:
                    print (f'No valid indices found for {name}.')
                    continue
                curr_start, curr_end = valid_idx[0], valid_idx[-1]
                total_frames += len(valid_idx)
                
                curr_motions = {}
                for idx in valid_idx:
                    curr_motion = curr_pkl[idx]
                    curr_motions[idx] = curr_motion
                curr_dict = {'video_name': name, 'hand_type': curr_hand_type, 'start': curr_start, 'end': curr_end, 'motions': curr_motions}
            
                rel_motion_data.append(curr_dict)
        
        self.valid_names = [x['video_name'] for x in rel_motion_data]
        print (f'Found {len(rel_motion_data)} relevant clips with {total_frames} valid frames.')
        return rel_motion_data

    def load_hand_masks(self, hand_mask_dir) -> None:
        all_hand_masks = glob(os.path.join(hand_mask_dir, '*hand_masks.pkl'))
        if len(all_hand_masks) == 0:
            all_hand_masks = os.listdir(hand_mask_dir) # each mask is a separate file for corresponding video frames
        self.hand_masks = {}
        for hand_mask in tqdm(all_hand_masks):
            if os.path.exists(hand_mask):
                video_name = hand_mask.split('/')[-1].split('_')[0]
            else:
                video_name = hand_mask.split('_')[0]
            if video_name in self.valid_names:
                # this does not scale, dummy dict created here
                # each mask is loaded separately later on
                # # load hand_masks as pickle
                # with open(hand_mask, 'rb') as f:
                #     curr_masks = pickle.load(f)
                # self.hand_masks[video_name] = curr_masks
                self.hand_masks[video_name] = {}
        # total_masks = sum([len(self.hand_masks[x]) for x in self.hand_masks])
        self.valid_names = list(self.hand_masks.keys())
        print (f'Found hand masks for {len(self.hand_masks)} valid videos from total {len(all_hand_masks)} processed videos.')
    
    def load_obj_masks(self, obj_mask_dir) -> None:
        all_obj_masks = glob(os.path.join(obj_mask_dir, '*obj_masks.pkl'))
        self.obj_masks = {}
        for obj_mask in tqdm(all_obj_masks):
            video_name = obj_mask.split('/')[-1].split('_')[0]
            if video_name in self.valid_names:
                # this does not scale, dummy dict created here
                # each mask is loaded separately later on
                # # load obj_masks as pickle
                # with open(obj_mask, 'rb') as f:
                #     curr_masks = pickle.load(f)
                # self.obj_masks[video_name] = curr_masks
                self.obj_masks[video_name] = {}
        # total_masks = sum([len(self.obj_masks[x]) for x in self.obj_masks])
        self.valid_names = list(self.obj_masks.keys())
        print (f'Found object masks for {len(self.obj_masks)} valid videos from total {len(all_obj_masks)} processed videos.')

    def load_action_data(self, action_path) -> None:
        action_pkls = glob(os.path.join(action_path, '*.pkl'))
        self.action_data = {}
        for pkl in tqdm(action_pkls):
            video_name = pkl.split('/')[-1].split('.')[0].replace('_action', '')
            if video_name in self.valid_names:
                with open(pkl, 'rb') as f:
                    c_data = pickle.load(f)
                    self.action_data[video_name] = c_data
    
    def process_actions(self) -> None:
        self.range2task = {}
        for video_name in self.action_data:
            self.range2task[video_name] = {}
            curr_actions = self.action_data[video_name]['actions']
            for act in curr_actions:
                st, end = act['start'], act['end']
                self.range2task[video_name][(st, end)] = act['verb'] + ' ' + act['noun']

    def find_superset(self, v_name, c_range):
        # find the superset of the given range in self.range2task
        all_ranges = list(self.range2task[v_name].keys())
        all_ranges = sorted(all_ranges, key=lambda x: x[0])
        for idx, (c_st, c_end) in enumerate(all_ranges):
            if c_st <= c_range[0] and c_end >= c_range[1]:
                return (c_st, c_end)
        return -1

    def process_motions(self) -> None:
        self.hand_type = self.args.hand_type
        all_joints, all_joints_cam, all_thetas, all_betas = [], [], [], []
        all_joints_ref, all_cam_t, all_rel_c2c = [], [], []
        all_names, all_ranges, all_tasks = [], [], []
        all_contacts, all_contact_maps = [], []
        for curr_data in tqdm(self.motion_data):
            if curr_data['video_name'] not in self.valid_names:
                continue
            curr_motion = self.action_data[curr_data['video_name']]
            c_joints, c_joints_cam, c_thetas, c_betas = [], [], [], []
            c_joints_ref, c_cam_t, c_rel_c2c = [], [], []
            all_ids = sorted(list(curr_data['motions'].keys()))
            if len(all_ids) == 0:
                continue
            ref_idx = all_ids[0]
            valid_ids = []
            for c_idx in curr_data['motions']: # keys are not contiguous, missing frames in between, pass these to the contact module below
                m_data = curr_data['motions'][c_idx]
                if self.hand_type not in m_data or len(m_data[self.hand_type]['joints']) == 0:
                    continue
                valid_ids.append(c_idx)
                curr_hand = m_data[self.hand_type]
                joints = curr_hand['joints']
                cam_t = curr_hand['cam_t']
                joints_cam = joints + cam_t[None]

                thetas = torch.from_numpy(curr_hand['thetas']).squeeze(0)
                thetas = rot_conv.matrix_to_axis_angle(thetas).numpy()
                betas = torch.from_numpy(curr_hand['betas']).squeeze(0).numpy()

                c_joints.append(joints)
                c_joints_cam.append(joints_cam)
                c_thetas.append(thetas)
                c_cam_t.append(cam_t)
                c_betas.append(betas)

                # compute relative camera pose between current and previous frame
                curr_c2w = curr_motion['pose'][c_idx]['cam2world'] # 4x4 camera to world transformation matrix SE(3)
                ref_c2w = curr_motion['pose'][ref_idx]['cam2world'] # 4x4 camera to world transformation matrix SE(3)
                rel_c2c = np.linalg.inv(np.linalg.inv(ref_c2w) @ curr_c2w) # relative camera pose
                c_rel_c2c.append(rel_c2c)

                # transform joints to reference frame
                inv_c2c = np.linalg.inv(rel_c2c)
                joints_cam_ref = joints_cam @ inv_c2c[:3, :3].T + inv_c2c[:3, 3]
                c_joints_ref.append(joints_cam_ref)

            if len(c_joints) > 0:
                all_joints.append(np.stack(c_joints, axis=0))
                all_joints_cam.append(np.stack(c_joints_cam, axis=0))
                all_thetas.append(np.stack(c_thetas, axis=0))
                all_betas.append(np.stack(c_betas, axis=0))

                all_joints_ref.append(np.stack(c_joints_ref, axis=0))
                all_cam_t.append(np.stack(c_cam_t, axis=0))
                all_rel_c2c.append(np.stack(c_rel_c2c, axis=0))

                all_names.append(curr_data['video_name'])
                all_ranges.append((curr_data['start'], curr_data['end']))

                ss_range = self.find_superset(curr_data['video_name'], (curr_data['start'], curr_data['end']))
                assert ss_range != -1, 'superset range not found'
                curr_task = self.range2task[curr_data['video_name']][ss_range]
                all_tasks.append(curr_task)

                # load object masks
                obj_mask_file = os.path.join(self.obj_mask_dir, f'{curr_data["video_name"]}_obj_masks.pkl')
                obj_masks = pickle.load(open(obj_mask_file, 'rb'))

                # fix length mismatch between motion (account for missed joints somehow) and contact
                c_contact, c_contact_mask, c_contact_map =  self.process_contacts(curr_data['video_name'], curr_data['motions'], 
                                        obj_masks, curr_data['start'], curr_data['end'], valid_ids, task='_'.join(curr_task.split(' ')))
                del obj_masks
                
                # set c_contact to nan if no contact found
                rel_contact = []
                for idx, con in enumerate(c_contact):
                    if len(con) == 0:
                        rel_contact.append(np.nan*np.ones((0, 3)))
                    else:
                        rel_contact.append(con)
                all_contacts.append(rel_contact)
                all_contact_maps.append(c_contact_map)
        
        assert len(all_joints) == len(all_joints_cam) == len(all_thetas) == len(all_betas) == len(all_joints_ref) == len(all_cam_t) == len(all_rel_c2c) == len(all_names) == len(all_ranges) == len(all_tasks) == len(all_contacts)
        all_motions = {'joints': all_joints, 'joints_cam': all_joints_cam, 'thetas': all_thetas, 'betas': all_betas,
                       'joints_ref': all_joints_ref, 'cam_t': all_cam_t, 'rel_c2c': all_rel_c2c,
                       'names': all_names, 'ranges': all_ranges, 'tasks': all_tasks,
                       'contacts': all_contacts, 'contact_maps': all_contact_maps}

        if len(all_names) == 0:
            print ('No valid motion sequences found.')
            return
        
        file = os.path.join(self.args.out_folder, f'contact_{self.args.start_num:04d}_{self.args.end_num:04d}_fix.pkl')
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        with open(file, 'wb') as f:
            pickle.dump(all_motions, f)

    def process_contacts(self, v_name, hand_masks, obj_masks, start, end, valid, task='') -> None:
        if self.args.use_gt_f:
            holo_focal = 685.88 # be careful about the focal length
        else:
            holo_focal = 17500
        im_w, im_h = (896, 504)
        intrx = np.array([[holo_focal, 0, im_w/2], [0, holo_focal, im_h/2], [0, 0, 1]])
        thresh = 10 # pixels threshold for determining 2D contact points

        if self.args.debug:
            random_save = False
            # save randomly with probability 0.1
            prob = 0.05
            if random.random() < prob:
                random_save = True
            if random_save:
                save_dir = os.path.join(self.args.out_folder, f'debug_contact/{v_name}_{start:06d}_{end:06d}_{task}')
                # args for renderer
                renderer_args = dict(
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color = (1.0, 1.0, 1.0),
                        focal_length = holo_focal, # 685.88 for holoassist
                        camera_z = 0.25, # check what works
                    )

        img_dir = os.path.join(self.args.img_folder, v_name, 'Export_py/Video/images_jpg')

        both_present, no_intersect = 0, 0

        all_hand_verts, all_contact_verts = [], []
        all_centers = []
        all_contact_maps = []
        # all_contact_masks = [0]*(end-start+1)
        all_contact_masks = [0]*len(valid)
        for idx, i in enumerate(valid):

            if i not in obj_masks or i not in hand_masks:
                # all_hand_verts.append(np.zeros((0, 3)))
                all_contact_verts.append(np.zeros((0, 3)))
                all_contact_maps.append(np.zeros((778,)))
                continue

            both_present += 1

            curr_hand_mask = None
            if 'mask' in hand_masks[i][self.hand_type]:
                # curr_hand_mask = hand_masks[i]['right']['mask']
                curr_hand_mask = np.array(hand_masks[i][self.hand_type]['mask'])
            elif 'verts' in hand_masks[i][self.hand_type]:
                curr_verts = hand_masks[i][self.hand_type]['verts'][0]
                curr_camt = hand_masks[i][self.hand_type]['cam_t'][0]
                curr_verts_cam = curr_verts + curr_camt[None,:]
                # project to image
                verts_cam = np.dot(intrx, curr_verts_cam.T).T
                verts_px = verts_cam[:, :2] / verts_cam[:, 2][:, None]
                hand_mask = create_mask_with_concave_hull(verts_px, (im_w, im_h), alpha=self.args.alpha)
                curr_hand_mask = np.array(hand_mask)
                if hand_mask is not None:
                    curr_hand_mask = curr_hand_mask > 0
                    hand_masks[i][self.hand_type]['mask'] = curr_hand_mask
            
            if curr_hand_mask is None:
                all_contact_verts.append(np.zeros((0, 3)))
                all_contact_maps.append(np.zeros((778,)))
                continue
            
            ####### could also load hand masks from hands23 model if concave hull fails ########
            # if curr_hand_mask is None:
            #     hand_mask_file = os.path.join(self.hand_mask_dir, f'{v_name}_hand_masks.pkl')
            #     if os.path.exists(hand_mask_file):
            #         loaded_masks = pickle.load(open(hand_mask_file, 'rb'))
            #         curr_hand_mask = loaded_masks[i][self.hand_type]['mask']
            #     else:
            #         hand_mask_file = os.path.join(self.hand_mask_dir, f'{v_name}_hand_masks/{i:06d}.npz')
            #         loaded_masks = np.load(hand_mask_file)['mask']
            #         curr_hand_mask = np.zeros((im_h, im_w))
            #         if 'right' in self.hand_type:
            #             val = 1
            #         else:
            #             val = 2
            #         curr_hand_mask[loaded_masks == val] = 255
            #         curr_hand_mask[loaded_masks == 3] = 255 # in case both hands are present at pixel, its value is 3
            #     if isinstance(curr_hand_mask, list) or isinstance(curr_hand_mask, Image.Image):
            #         curr_hand_mask = np.array(curr_hand_mask)
            #     hand_masks[i][self.hand_type]['mask'] = curr_hand_mask
            #     del loaded_masks
            
            curr_obj_mask = obj_masks[i]['mask']
            if isinstance(curr_obj_mask, list) or isinstance(curr_obj_mask, Image.Image):
                curr_obj_mask = np.array(curr_obj_mask)

            curr_hand_mask = pad_mask_at_boundary(curr_hand_mask, padding=2)
            if curr_hand_mask is None:
                all_contact_verts.append(np.zeros((0, 3)))
                all_contact_maps.append(np.zeros((778,)))
                continue
            curr_obj_mask = pad_mask_at_boundary(curr_obj_mask, padding=2)
            if curr_obj_mask is None:
                all_contact_verts.append(np.zeros((0, 3)))
                all_contact_maps.append(np.zeros((778,)))
                continue

            hand_box = None
            if 'box' in hand_masks[i][self.hand_type]:
                # get hand and obj boxes
                hand_box = hand_masks[i][self.hand_type]['box']
            elif 'mask' in hand_masks[i][self.hand_type]:
                hand_box = get_bbox_from_mask(curr_hand_mask)
            
            if hand_box is None:
                all_contact_verts.append(np.zeros((0, 3)))
                all_contact_maps.append(np.zeros((778,)))
                continue
            
            obj_box = obj_masks[i]['box']

            # Compute intersection of hand and obj masks
            intersection_mask = np.logical_and(curr_hand_mask, curr_obj_mask)
            mask_indices = np.where(intersection_mask > 0)

            # consider hand mask in the intersection area
            hand_mask = np.logical_and(curr_hand_mask, intersection_mask)
            mask_indices = np.where(hand_mask > 0)
            
            if len(mask_indices[0]) == 0:
                no_intersect += 1
                all_contact_verts.append(np.zeros((0, 3)))
                all_contact_maps.append(np.zeros((778,)))
                continue
            else:
                x0, x1 = mask_indices[1].min(), mask_indices[1].max()
                y0, y1 = mask_indices[0].min(), mask_indices[0].max()
                intersect = [x0, y0, x1, y1]
                both_x, both_y = mask_indices[1], mask_indices[0]

                # center of intersect
                c_width, c_height = (intersect[0]+intersect[2])/2, (intersect[1]+intersect[3])/2
            
            all_centers.append((c_width, c_height))
            
            # repeated from above
            curr_verts = hand_masks[i][self.hand_type]['verts'][0]
            curr_camt = hand_masks[i][self.hand_type]['cam_t'][0]
            curr_verts_cam = curr_verts + curr_camt[None,:]
            # project to image
            verts_cam = np.dot(intrx, curr_verts_cam.T).T
            verts_px = verts_cam[:, :2] / verts_cam[:, 2][:, None]
            
            curr_boths = np.stack([both_x, both_y], axis=1)

            # find closes vert to center within a threshold
            v_dists = np.linalg.norm(verts_px[:,None,:] - curr_boths[None], axis=-1)
            dists = np.min(v_dists, axis=1)
            close_dists = dists < thresh
            close_verts_px = verts_px[close_dists]
            close_verts_ids = np.where(close_dists)[0]
            close_verts_3d = curr_verts_cam[close_verts_ids]
            
            all_contact_verts.append(close_verts_3d)
            all_contact_masks[idx] = 1
            
            # one hot contact map
            contact_map = np.zeros((778,))
            contact_map[close_verts_ids] = 1
            all_contact_maps.append(contact_map)

            if self.args.debug and random_save:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                ######### for visualization and debugging #########
                # visualize hand mesh and contact maps
                curr_args = renderer_args.copy()
                curr_args.update({'contact_vertices': contact_map})
                render_hand = self.renderer.render_rgba_contact(curr_verts_cam, **curr_args)
                posed_img = Image.fromarray((render_hand[..., :3] * 255).astype('uint8'))

                # render a flat hand to show the contact map only
                rot_mat_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
                global_orient = torch.from_numpy(rot_mat_y).float().view(1, 1, 3, 3) # this followed by a 90 degree rotation around x-axis will make the hand vertical
                hand_pose = torch.eye(3).view(1,1,3,3).repeat(1,15,1,1)
                betas = torch.zeros(1, 10)
                mano_output = self.mano_right(global_orient=global_orient, hand_pose=hand_pose, betas=betas, pose2rot=False)
                mano_verts = mano_output.vertices[0].detach()
                mano_verts = mano_verts - mano_verts.mean(dim=0, keepdim=True)
                curr_args.update({'rot_angle': 90, 'rot_axis': (1, 0, 0)})
                render_img = self.renderer.render_rgba_contact(mano_verts.numpy(), **curr_args)
                flat_img = Image.fromarray((render_img[..., :3] * 255).astype(np.uint8))

                img_path = os.path.join(img_dir, f'{i:06d}.jpg')
                raw_image = Image.open(img_path)

                # concatenate the images
                concat_img = Image.new('RGB', (posed_img.width + flat_img.width, posed_img.height))
                concat_img.paste(posed_img, (0, 0))
                concat_img.paste(flat_img, (posed_img.width, 0))
                concat_img.save(os.path.join(save_dir, f'render_{i:06d}.jpg'))

                # plot verts_px on raw_image and create a mask out of projected verts
                curr_img = raw_image.copy()
                draw = ImageDraw.Draw(curr_img)
                pw = 2
                for vert in verts_px:
                    draw.ellipse([vert[0]-pw, vert[1]-pw, vert[0]+pw, vert[1]+pw], fill='red')
                # save curr_img with verts
                curr_img.save(os.path.join(save_dir, f'verts_{i:06d}.jpg'))

                # # take convex hull of all verts_px
                # hull_mask = create_mask_with_convex_hull(verts_px, raw_image.size)
                # # alpha blend hull_mask with raw_image
                # curr_img = blend_mask_with_image(hull_mask, raw_image, alpha=128)
                # curr_img.save(os.path.join(save_dir, f'hull_{i:06d}.jpg'))

                # # create a polygon outline of 2D pixels verts_px
                # curr_img = create_mask_with_polygon_outline(verts_px, raw_image.size)
                # # blend curr_img with raw_image
                # curr_img = blend_mask_with_image(curr_img, raw_image, alpha=128)
                # curr_img.save(os.path.join(save_dir, f'poly_{i:06d}.jpg'))

                # # Compute the alpha shape (concave hull), this works better than convex hull or polygon outline
                # mask = create_mask_with_concave_hull(verts_px, raw_image.size)
                
                # alpha blend hand_mask with curr_img
                mask = blend_mask_with_image(curr_hand_mask, raw_image, alpha=128)
                mask.save(os.path.join(save_dir, f'concave_{i:06d}.jpg'))

                # alpha blend raw_image with curr_hand_mask and curr_obj_mask
                hand_mask_alpha = Image.fromarray((curr_hand_mask * 255).astype(np.uint8))
                obj_mask_alpha = Image.fromarray((curr_obj_mask * 255).astype(np.uint8))
                hand_mask_alpha.putalpha(128)
                obj_mask_alpha.putalpha(128)
                mask_image = raw_image.copy()
                mask_image.paste(hand_mask_alpha, (0, 0), hand_mask_alpha)
                mask_image.paste(obj_mask_alpha, (0, 0), obj_mask_alpha)
                mask_image.save(os.path.join(save_dir, f'masks_{i:06d}.jpg'))

                # plot contact points on raw_image
                draw = ImageDraw.Draw(raw_image.copy())

                # draw hand and obj boxes
                # print (hand_box, obj_box)
                draw.rectangle(np.array(hand_box).tolist(), outline='red', width=5)
                draw.rectangle(np.array(obj_box).tolist(), outline='blue', width=5)

                # plot verts_cam
                pw = 2
                for vert in close_verts_px:
                    draw.ellipse([vert[0]-pw, vert[1]-pw, vert[0]+pw, vert[1]+pw], fill='cyan')
                
                # # draw contact points bounding box
                # draw.rectangle(intersect, outline='green')

                print (len(both_x), len(both_y), len(close_verts_3d))

                # create a gaussian mask around the center
                sigma = 10
                x = np.arange(0, raw_image.size[0], 1, float)
                y = np.arange(0, raw_image.size[1], 1, float)
                x, y = np.meshgrid(x, y)
                # create a gaussian mask
                gauss = np.exp(-((x - c_width)**2 + (y - c_height)**2) / (2.0 * sigma**2))

                prev_center = [c_width, c_height]

                # Convert the original image to RGBA (if not already) to support transparency
                final_image = raw_image.convert('RGBA')

                # compute gaussian mask for each for in both_x, both_y
                for c_x, c_y in zip(both_x, both_y):
                    overlay_image = create_gussian_mask((c_x, c_y), 2, raw_image.size[0], raw_image.size[1])
                    # Composite the overlay onto the original image
                    final_image = Image.alpha_composite(final_image, overlay_image)
                # Convert back to RGB to display/save if necessary
                final_image = final_image.convert('RGB')

                # draw = ImageDraw.Draw(final_image)
                # # draw mask_indices on final_image
                # for j in range(len(mask_indices[0])):
                #     draw.point((mask_indices[1][j], mask_indices[0][j]), fill='green')

                # save final_image
                final_image.save(os.path.join(save_dir, f'contact_{i:06d}.jpg'))

                # plt.close()
                # plt.imshow(final_image)
                # plt.axis('off')
                # plt.show()
                # break

        print (f'Processing video {v_name}_{start:06d}_{end:06d}_{task}', 'Found hand & object masks:', both_present, 'No intersect:', no_intersect, 'Total:', len(all_contact_verts))
        return all_contact_verts, all_contact_masks, all_contact_maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate processed videos in one place')
    parser.add_argument('--img_folder', type=str, default=None, help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save rendered results')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample clips from the video')
    parser.add_argument('--task_type', nargs='+', default=[''], help='Task type to process')
    parser.add_argument('--start_num', type=int, default=None, help='start video index')
    parser.add_argument('--end_num', type=int, default=None, help='end video index')
    parser.add_argument('--use_gt_f', action='store_true', default=False, help='Use ground truth focal length')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value for concave hull')
    parser.add_argument('--hand_type', type=str, default='right', help='Hand type to process, only right hand considered for now')
    args = parser.parse_args()

    reset_all_seeds(seed=42)

    if args.img_folder is None:
        args.img_folder = os.environ['HOLO_PATH']
    assert os.path.exists(args.img_folder), f'Path {args.img_folder} does not exist.'

    dataset = HoloContacts(args)
    
    print (len(dataset))