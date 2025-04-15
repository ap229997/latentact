import sys
import os
import glob
from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from PIL import Image
from utils.utils import center_and_resize
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as transforms
import random
import argparse
import pickle
import codecs as cs
import json
import common.rotation_conversions as rot


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def get_dataset(name, indices=False):
    if not indices:
        DATASETS = {
            'holo': HoloMotion,
            'arctic': ArcticMotion
        }
    else:
        DATASETS = {
            'holo': CodebookIndicesData,
            'arctic': ArcticCodebookIndicesData,
        }

    return DATASETS[name]


class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = [] # list of sequences
        self.lengths = [] # length of each sequence
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)


class HoloMotion(data.Dataset):
    def __init__(self, opt, split='train'):
        if opt is not None:
            self.opt = opt
        else:
            self.opt = argparse.Namespace()
            self.opt.window_size = 10
            self.opt.joints_num = 21
        self.split = split     
        
        # load contact data from preprocessed files
        # useful to avoid too much computation during data loading
        self.contact_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_contact_dir')
        data = self.load_contacts(self.contact_dir)

        # run the check once when contact data is preprocessed again
        # verify = self.sanity_check(data)
        # if not verify:
        #     raise ValueError('Data is not consistent with the transformation matrix')

        if self.opt.setting is not None:
            data = self.load_setting(data, split)
        else:
            if not self.opt.return_indices:
                data = self.create_splits(data, split)
        
        if 'joints' in self.opt.motion_type:
            self.motion = data[self.opt.motion_type]
            self.motion = [m.reshape(len(m), -1) for m in self.motion]
            self.mean = np.mean(np.concatenate(self.motion), axis=0)
            self.std = np.std(np.concatenate(self.motion), axis=0)
        elif 'mano' in self.opt.motion_type:
            self.thetas = data['thetas']
            self.thetas = [m.reshape(len(m), -1) for m in self.thetas]
            self.thetas_mean = np.zeros_like(self.thetas[0][0])
            self.thetas_std = np.ones_like(self.thetas[0][0])
            self.betas = data['betas']
            self.betas = [m.reshape(len(m), -1) for m in self.betas]
            self.betas_mean = np.zeros_like(self.betas[0][0])
            self.betas_std = np.ones_like(self.betas[0][0])
            self.cam_t = data['cam_t']
            self.cam_t = [m.reshape(len(m), -1) for m in self.cam_t]
            self.cam_t_mean = np.mean(np.concatenate(self.cam_t), axis=0) # these are zerod out later, too many settings so fixed values not used yet
            self.cam_t_std = np.std(np.concatenate(self.cam_t), axis=0) # these are replaced with ones later, too many settings so fixed values not used yet
            assert len(self.thetas) == len(self.betas) == len(self.cam_t)
            # concatenate all
            self.mano = [np.concatenate([self.thetas[i], self.betas[i], self.cam_t[i]], axis=-1) for i in range(len(self.thetas))]
            self.mean = np.concatenate([self.thetas_mean, self.betas_mean, self.cam_t_mean], axis=-1)
            self.std = np.concatenate([self.thetas_std, self.betas_std, self.cam_t_std], axis=-1)
        else:
            raise ValueError('Unknown motion type')
        
        # load precomputed video features
        if self.opt.video_feats is not None:
            if self.opt.use_inpaint:
                feats_path = os.path.join(os.environ["ROOT_DIR"], 'downloads/holo_video_inpaint_feats')
            else:
                feats_path = os.path.join(os.environ["ROOT_DIR"], 'downloads/holo_video_feats')

            feats = self.load_features(feats_path)

            if self.opt.video_feats == 2048:
                feat_prefix = 'conv_'
            elif self.opt.video_feats == 768:
                feat_prefix = 'tf_'

            vid_feats = []
            for idx in range(len(data['names'])):
                name, (st, end) = data['names'][idx], data['ranges'][idx]
                vid_feats.append(feats[name][(st, end)][f'{feat_prefix}feat'])
            assert len(vid_feats) == len(data['names'])
            self.feats = []

        # load precomputed text features
        if self.opt.text_feats:
            self.text_feats = []
            if self.opt.video_feats is None:
                if self.use_inpaint:
                    feats_path = os.environ["VIDEO_INPAINT_FEATS_DIR"] # already included in the video features file
                else:
                    feats_path = os.environ["VIDEO_FEATS_DIR"]
            
            self.texts = []
            for idx in range(len(data['names'])):
                name, (st, end) = data['names'][idx], data['ranges'][idx]
                self.texts.append(feats[name][(st, end)]['text_feat'])
            
            assert len(self.texts) == len(data['names'])

        if self.opt.contact_grid is not None:
            self.contacts = []

        if self.opt.contact_map:
            self.contact_maps = []

        data_joints_ref = data['joints_ref']
        data_joints_ref = [m.reshape(len(m), -1) for m in data_joints_ref]
        self.joints_ref = []

        self.data = [] # list of sequences
        self.lengths = [] # length of each sequence
        self.names = [] # video name
        self.ranges = [] # start, end of each sequence in the video
        self.tasks = [] # task for each sequence

        relevant_list = self.motion if 'joints' in self.opt.motion_type else self.mano
        relevant_indices = []
        for j, curr_motion in enumerate(relevant_list):
            if curr_motion.shape[0] < self.opt.window_size:
                continue
            
            # filter based on self.cam_t_mean and self.cam_t_std
            # aggressive filtering to bad data, training doesn't work well without this
            if 'mano' in self.opt.motion_type:
                if np.any(np.abs(self.cam_t[j] - self.cam_t_mean[None]) > self.cam_t_std[None]):
                    continue

            relevant_indices.append(j)
            self.lengths.append(curr_motion.shape[0] - self.opt.window_size)
            self.data.append(curr_motion)
            self.names.append(data['names'][j])
            self.ranges.append(data['ranges'][j])
            self.tasks.append(data['tasks'][j])
            if self.opt.video_feats is not None:
                self.feats.append(vid_feats[j])
            if self.opt.text_feats:
                self.text_feats.append(self.texts[j])
            if self.opt.contact_grid is not None:
                self.contacts.append(data['contacts'][j])
            self.joints_ref.append(data_joints_ref[j])
            if self.opt.contact_map:
                self.contact_maps.append(data['contact_maps'][j])

        self.rel_c2c = [data['rel_c2c'][i] for i in relevant_indices]

        if opt.contact_grid is not None:
            # precomputed contact volumes are loaded if available, else compute them
            self.compute_contact_volumes()
        
        # set self.cam_transf_mask to all ones
        self.cam_transf_mask = [np.ones(len(self.data[i])) for i in range(len(self.data))]

        self.reset_mean_std() # mean and std reset here, too many settings so fixed values not used yet

        # set joint_ref_mask to all ones
        self.joints_ref_mask = [np.ones(len(self.data[i])) for i in range(len(self.data))]
        self.update_joints_ref_mask()

        if self.opt.pred_cam:
            # compute camera translation and rotation w.r.t. reference frame
            self.compute_camera_transformation(data['rel_c2c'], relevant_indices)

            # append camera transformation matrix to the end of the self.data
            self.data = [np.concatenate([self.data[i], self.cam_rot[i]], axis=-1) for i in range(len(self.data))] # (T, 58+3+3+3) -> (T, 58+3+3+3+6)
            self.mean = np.concatenate([self.mean, np.zeros(6)], axis=-1)
            self.std = np.concatenate([self.std, np.ones(6)], axis=-1)

            self.data = [np.concatenate([self.data[i], self.cam_transl[i]], axis=-1) for i in range(len(self.data))] # (T, 58+3+3+3+6) -> (T, 58+3+3+3+6+3)
            self.mean = np.concatenate([self.mean, np.zeros(3)], axis=-1)
            self.std = np.concatenate([self.std, np.ones(3)], axis=-1)
        
        if self.opt.decoder_only and self.opt.load_indices is not None:
            # load codebook indices for training the predictor module
            indices_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.load_indices, 'model')
            prefix = ''
            if self.opt.transfer_from is not None:
                transfer_dataset, transfer_model = self.opt.transfer_from.split('/')
                prefix = f'transfer_{transfer_dataset}_{transfer_model}_'
            indices = glob.glob(f'{indices_dir}/{prefix}*finest*{self.split}*.pkl')
            if self.opt.transfer_from is None:
                # remove any entry with 'transfer' in it
                indices = [i for i in indices if 'transfer' not in i]

            indices.sort()
            if len(indices) == 0:
                indices = glob.glob(f'{indices_dir}/*{self.split}*.pkl')
                indices.sort()
                if len(indices) == 0:
                    print (f'No indices file found in {indices_dir}, Exiting...')
                    sys.exit()
            indices_file = indices[-1]
            self.indices = pickle.load(open(indices_file, 'rb'))
        
        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def load_setting(self, data, split):
        settings_file = os.path.join(os.environ['ROOT_DIR'], 'data/settings.json')
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        curr_setting = settings[self.opt.setting]
        split_ids = curr_setting[split]
        # split data
        data = {k: [v[i] for i in split_ids] for k, v in data.items()}
        print (f'Loaded {len(data["names"])} sequences for setting {self.opt.setting} in {split} split')
        return data
    
    def create_splits(self, data, split):
        if os.path.isdir(self.contact_dir): # only do if contact data is precomputed
            splits_file = os.path.join(os.environ["ROOT_DIR"], 'data/splits.json')
            if not os.path.exists(splits_file):
                val_ids = random.sample(range(len(data['names'])), int(0.2*len(data['names'])))
                train_ids = [i for i in range(len(data['names'])) if i not in val_ids]
                splits = {'train': train_ids, 'val': val_ids}
                # save in file
                with open(splits_file, 'w') as f:
                    json.dump(splits, f)
            else:
                with open(splits_file, 'r') as f:
                    splits = json.load(f)
                split_ids = splits[split]
        else:
            raise ValueError('Precompute contact data first before creating splits')

        # split data
        data = {k: [v[i] for i in split_ids] for k, v in data.items()}

        return data

    def load_features(self, feat_dir):
        files = glob.glob(f'{feat_dir}/*feat*.pkl')
        feat_data = {}
        # combine dicts from all pkl files
        for file in tqdm(files):
            with open(file, 'rb') as f:
                data = pickle.load(f) # dict of dicts
                for k, v in data.items():
                    if k not in feat_data:
                        feat_data[k] = v
                    else:
                        feat_data[k].update(v)
        return feat_data
    
    def load_contacts(self, contact_dir):
        files = glob.glob(f'{contact_dir}/contact_*_fix.pkl')
        contact_data = {}
        # combine dicts from all pkl files
        for file in tqdm(files):
            st, end = int(file.split('_')[-3]), int(file.split('_')[-2])
            with open(file, 'rb') as f:
                data = pickle.load(f)
                for k, v in data.items():
                    if k not in contact_data:
                        contact_data[k] = v
                    else:
                        contact_data[k] += v
        num_seq = len(contact_data['names'])
        print (f'Loaded {num_seq} sequences with contact')
        return contact_data
    
    def sanity_check(self, c_data):
        all_diffs = []
        jcs = c_data['joints_cam'] # (N, T, 1, 21, 3)
        jrs = c_data['joints_ref'] # (N, T, 1, 21, 3)
        jts = c_data['rel_c2c'] # (N, T, 4, 4)
        print (f'Running sanity check on {len(jcs)} sequences')
        for i in tqdm(range(len(jcs))):
            c_jc = jcs[i][:,0] # (T, 21, 3)
            c_jr = jrs[i][:,0] # (T, 21, 3)
            c_jt = jts[i] # (T, 4, 4)
            for j in range(len(c_jc)):
                jc = c_jc[j]
                jr = c_jr[j]
                jt = c_jt[j]
                jt_inv = np.linalg.inv(jt)
                jr_ = jc @ jt_inv[:3,:3].T + jt_inv[:3,3]
                diff = np.linalg.norm(jr - jr_, axis=1).mean()
                all_diffs.append(diff)
                if not (diff < 3e-3 and np.allclose(jr, jr_)):
                    return False
        # print (np.mean(all_diffs), np.min(all_diffs), np.max(all_diffs))
        return True
    
    def inv_transform(self, data):
        if 'joints' in self.opt.motion_type:
            out = data * self.std + self.mean
        elif 'mano' in self.opt.motion_type:
            out = data
        return out

    def __len__(self):
        if self.opt.debug:
            if self.opt.viz_gt or self.opt.viz_pred is not None:
                return min(2000, self.cumsum[-1])
            else:
                return min(100, self.cumsum[-1])
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        if 'joints' in self.opt.motion_type:
            "Z Normalization"
            motion = (motion - self.mean) / self.std
        elif 'mano' in self.opt.motion_type:
            motion = (motion - self.mean) / self.std
            
        batch = {'motion': motion, 'length': motion.shape[0], 'name': self.names[motion_id], 'range': self.ranges[motion_id],
                 'start': idx, 'end': idx + self.opt.window_size, 'motion_id': motion_id, 'task': self.tasks[motion_id]}

        if self.opt.video_feats:
            curr_feats = self.feats[motion_id][idx:idx + self.opt.window_size]
            if self.opt.only_first:
                curr_feats = np.repeat(curr_feats[:1], self.opt.window_size, axis=0)
            elif self.opt.interpolate:
                curr_feats = np.repeat(curr_feats[:1] + curr_feats[-1:], self.opt.window_size, axis=0)
            batch['video_feats'] = curr_feats

        if self.opt.text_feats:
            curr_text_feats = self.text_feats[motion_id][idx:idx + self.opt.window_size]
            if self.opt.only_first or self.opt.interpolate:
                curr_text_feats = np.repeat(curr_text_feats[:1], self.opt.window_size, axis=0)
            batch['text_feats'] = curr_text_feats

        if self.opt.contact_grid is not None:
            curr_contact_centroid_mask = self.contact_centroids_mask[motion_id][idx:idx + self.opt.window_size]

            curr_contact = self.contact_volumes[motion_id][idx:idx + self.opt.window_size]
            curr_contact_mask = self.contact_masks[motion_id][idx:idx + self.opt.window_size]
            if self.opt.only_first:
                curr_contact = np.repeat(curr_contact[:1], self.opt.window_size, axis=0)
            
            curr_grid = self.grid_volumes[motion_id]
            curr_grid_mask = self.grid_masks[motion_id]
            batch['contact'] = curr_contact
            batch['contact_mask'] = curr_contact_mask * curr_contact_centroid_mask
            batch['grid'] = curr_grid
            batch['grid_mask'] = curr_grid_mask

            # check if this breaks the code somewhere
            batch['contact_point'] = self.contact_centroids[motion_id][idx:idx + self.opt.window_size] # required for visualization at inference time

        curr_rel_c2c = self.rel_c2c[motion_id][idx:idx + self.opt.window_size]
        batch['rel_c2c'] = curr_rel_c2c
        
        batch['cam_mask'] = self.cam_transf_mask[motion_id][idx:idx + self.opt.window_size]

        curr_joints_ref, curr_joints_ref_mask = self.compute_joints_ref(motion_id, idx)
        curr_joints_ref = curr_joints_ref.reshape(-1, self.opt.joints_num*3)

        # self.verify_joints_ref(curr_joints_ref, motion_id, idx) # verify if the joints_ref are correct

        # change self.cam_rot and self.cam_transl to self.cam_rot_rel and self.cam_transl_rel
        cam_rot_ref, cam_transl_ref, cam_transf_mask = self.compute_camera_transf_ref(motion_id, idx)
        batch['cam_rot_ref'] = cam_rot_ref
        batch['cam_transl_ref'] = cam_transl_ref
        if self.opt.pred_cam:
            batch['motion'][..., -9:-3] = cam_rot_ref
            batch['motion'][..., -3:] = cam_transl_ref
            batch['cam_mask'] = cam_transf_mask
        
        if self.opt.contact_grid is not None:
            curr_joints_ref_mask = curr_joints_ref_mask * curr_contact_centroid_mask
        batch['joints_ref'] = curr_joints_ref
        batch['joints_ref_mask'] = curr_joints_ref_mask

        if self.opt.contact_map:
            curr_contact_map = self.contact_maps[motion_id][idx:idx + self.opt.window_size]
            batch['contact_map'] = np.array(curr_contact_map).astype(np.int8) # (T, 778) one-hot encoding of contact map on hand vertices

        if self.opt.decoder_only and self.opt.load_indices is not None:
            code_idx = self.indices[self.names[motion_id]][f'{idx}_{idx+self.opt.window_size}']['indices']
            batch['code_idx'] = code_idx
        
        return batch
    
    def compute_contact_transl_ref(self, motion_id, ref_idx):
        cam_t_contact_ref = []
        cam_t_contact_ref_mask = []
        rel_cam_t_contact = self.contact_transl_rel[motion_id][ref_idx: ref_idx + self.opt.window_size] # in "world" frame
        # add the contact centroid of "world" frame to get to camera frame
        contact_centroid_world = self.contact_centroids[motion_id][0] # in camera frame
        contact_centroid_ref = self.contact_centroids[motion_id][ref_idx] # in camera frame
        ref_transf = self.rel_c2c[motion_id][ref_idx] # world to ref transf
        for i in range(rel_cam_t_contact.shape[0]):
            curr_cam_t_contact = rel_cam_t_contact[i] # in "world" frame
            curr_cam_t_contact += contact_centroid_world # in camera frame
            # convert to camera coordinate system at current frame
            curr_cam_t_contact = np.concatenate([curr_cam_t_contact, [1]], axis=-1)
            curr_cam_t_contact_cam_ref = ref_transf @ curr_cam_t_contact # in cam frame
            # cam to ref_idx frame
            curr_cam_t_contact_ref = curr_cam_t_contact_cam_ref[:3] - contact_centroid_ref
            cam_t_contact_ref.append(curr_cam_t_contact_ref)

            curr_mask = 1           
            if np.isnan(curr_cam_t_contact_ref).any():
                curr_mask = 0
            curr_mask *= self.contact_centroids_mask[motion_id][0]
            curr_mask *= self.contact_centroids_mask[motion_id][ref_idx]
            curr_mask *= self.contact_transl_rel_mask[motion_id][i]
            curr_mask *= self.cam_transf_mask[motion_id][ref_idx]

            cam_t_contact_ref_mask.append(curr_mask)
        
        cam_t_contact_ref = np.array(cam_t_contact_ref)
        cam_t_contact_ref_mask = np.array(cam_t_contact_ref_mask)

        return cam_t_contact_ref, cam_t_contact_ref_mask
    
    def compute_joints_ref(self, motion_id, ref_idx):
        joints_ref = []
        joints_ref_mask = []
        ref_transf = self.rel_c2c[motion_id][ref_idx]
        if self.opt.coord_sys == 'contact':
            contact_centroid_ref = self.contact_centroids[motion_id][ref_idx] # in camera frame
        rel_joints_ref = self.joints_ref[motion_id][ref_idx: ref_idx + self.opt.window_size] # in "world" frame
        for i in range(len(rel_joints_ref)):
            curr_joints_ref = rel_joints_ref[i].reshape(self.opt.joints_num, 3)
            # convert to camera coordinate system at current frame
            curr_joints_ref = np.concatenate([curr_joints_ref, np.ones((curr_joints_ref.shape[0], 1))], axis=-1)
            curr_joints_ref = (ref_transf @ curr_joints_ref.T).T[..., :3]
            if self.opt.coord_sys == 'contact':
                curr_joints_ref -= contact_centroid_ref[None]

            joints_ref.append(curr_joints_ref)

            curr_mask = 1
            if np.isnan(curr_joints_ref).any():
                curr_mask = 0
            if self.opt.coord_sys == 'contact':
                curr_mask *= self.contact_centroids_mask[motion_id][ref_idx]
            curr_mask *= self.joints_ref_mask[motion_id][i]
            curr_mask *= self.cam_transf_mask[motion_id][ref_idx]

            joints_ref_mask.append(curr_mask)

        joints_ref = np.array(joints_ref)
        joints_ref_mask = np.array(joints_ref_mask)

        return joints_ref, joints_ref_mask
    
    def verify_joints_ref(self, c_joints_ref, motion_id, ref_idx):
        rel_transf = self.rel_c2c[motion_id][ref_idx: ref_idx + self.opt.window_size]
        ref2world_transf = np.linalg.inv(rel_transf[0])
        for i in range(len(c_joints_ref)):
            js = c_joints_ref[i].reshape(self.opt.joints_num, 3) # cam ref_idx frame
            js_ref = self.joints_ref[motion_id][ref_idx + i].reshape(self.opt.joints_num, 3)
            # c_transf = rel_transf[i]
            # js_cam = np.concatenate([js_ref, np.ones((self.opt.joints_num, 1))], axis=-1)
            # js_cam = (c_transf @ js_cam.T).T[..., :3] # cam i frame
            # assert np.allclose(js, js_cam)

            js_world = np.concatenate([js, np.ones((self.opt.joints_num, 1))], axis=-1)
            js_world = (ref2world_transf @ js_world.T).T[..., :3]
            assert np.allclose(js_world, js_ref)
    
    def compute_camera_transf_ref(self, motion_id, ref_idx):
        cam_rot_ref = []
        cam_transl_ref = []
        cam_transf_mask = []
        rel_transf = self.rel_c2c[motion_id][ref_idx: ref_idx + self.opt.window_size]
        ref_transf = self.rel_c2c[motion_id][ref_idx]
        for i in range(len(rel_transf)):
            curr_transf = rel_transf[i]
            curr2ref_transf = ref_transf @ np.linalg.inv(curr_transf)
            # rotation
            curr2ref_rot = curr2ref_transf[:3,:3]
            curr2ref_rot = curr2ref_rot[:2,:3].flatten()
            cam_rot_ref.append(curr2ref_rot)

            # translation
            curr2ref_transl = curr2ref_transf[:3,3]
            cam_transl_ref.append(curr2ref_transl)

            curr_mask = 1
            if np.isnan(curr2ref_rot).any() or np.isnan(curr2ref_transl).any():
                curr_mask = 0
            curr_mask *= self.cam_transf_mask[motion_id][i]
            curr_mask *= self.cam_transf_mask[motion_id][ref_idx]
            
            cam_transf_mask.append(curr_mask)

        cam_rot_ref = np.array(cam_rot_ref)
        cam_transl_ref = np.array(cam_transl_ref)
        cam_transf_mask = np.array(cam_transf_mask)

        return cam_rot_ref, cam_transl_ref, cam_transf_mask

    
    def reset_mean_std(self):
        self.cam_t_mean = np.zeros(3)
        self.cam_t_std = np.ones(3)
        self.mean = np.zeros_like(self.mean)
        self.std = np.ones_like(self.std)
    
    def update_mean_std(self):
        # update mean and std of self.data cam_t term 58:61
        curr_cam_t = [self.data[i][..., 58:61] for i in range(len(self.data))]
        self.cam_t_mean = np.mean(np.concatenate(curr_cam_t), axis=0)
        self.cam_t_std = np.std(np.concatenate(curr_cam_t), axis=0)
        self.mean[..., 58:61] = self.cam_t_mean
        self.std[..., 58:61] = self.cam_t_std

    def update_contact_centroids_mask(self):
        for i in range(len(self.contact_centroids)):
            curr_contact = self.contact_centroids[i]
            curr_mask = np.ones(len(curr_contact))
            for j in range(len(curr_contact)):
                if np.isnan(curr_contact[j]).any():
                    curr_mask[j] = 0
            self.contact_centroids_mask[i] = curr_mask

    def update_joints_ref_mask(self):
        self.joints_ref_mask = []
        for i in range(len(self.joints_ref)):
            curr_ref = self.joints_ref[i]
            curr_mask = np.ones(len(curr_ref)) * self.contact_centroids_mask[i]
            for j in range(len(curr_ref)):
                if np.isnan(curr_ref[j]).any():
                    curr_mask[j] = 0
            self.joints_ref_mask.append(curr_mask)

    def compute_residual_joints_ref(self, joints_ref, rel_transf):
        residual_joints_ref = []
        for i in range(len(joints_ref)):
            c_ref = []
            for j in range(len(joints_ref[i])):
                prev_idx = max(0, j-1)
                prev_transf = rel_transf[i][prev_idx]
                c_joints_ref = joints_ref[i][j].reshape(self.opt.joints_num, 3)
                # apply prev_transf to c_joints_ref
                c_joints_ref = np.concatenate([c_joints_ref, np.ones((self.opt.joints_num, 1))], axis=-1)
                new_joints_ref = (prev_transf @ c_joints_ref.T).T
                c_ref.append(new_joints_ref[...,:3].reshape(-1))
            c_ref = np.array(c_ref, dtype=joints_ref[i].dtype)
            residual_joints_ref.append(c_ref)
        return residual_joints_ref
    
    def transform_global_params(self, rel_transf, rel_indices):
        self.cam_transf_mask = []
        for i in range(len(rel_indices)):
            c_transf_mask = np.ones(len(rel_transf[rel_indices[i]]))
            for j in range(len(rel_transf[rel_indices[i]])):
                curr_transf = rel_transf[rel_indices[i]][j]
                curr_inv_transf = np.linalg.inv(curr_transf)
                if np.isnan(curr_inv_transf).any():
                    c_transf_mask[j] = 0
                    continue

                mano_rot = self.data[i][j,:3] # (1, 3) rotation in axis-angle
                # rotate global orientation by inverse of relative transformation
                rot_matrix =  rot.axis_angle_to_matrix(torch.from_numpy(mano_rot)[None]).squeeze().numpy()
                new_rot_matrix = np.einsum('ij,aj->ai', curr_inv_transf[:3,:3], rot_matrix)
                new_rot = rot.matrix_to_axis_angle(torch.from_numpy(new_rot_matrix)[None]).squeeze().numpy()
                self.data[i][j,:3] = new_rot

                # transform global translation by inverse of relative transformation
                mano_transl = self.data[i][j,58:61] # (1, 3) translation
                # convert to homogeneous coordinates
                mano_transl = np.concatenate([mano_transl, [1]], axis=-1) # (1, 4)
                new_transl = np.einsum('ij,aj->ai', curr_inv_transf, mano_transl[None])
                self.data[i][j,58:61] = new_transl[0,:3]

            self.cam_transf_mask.append(c_transf_mask)

    def compute_residual_transformation(self, rel_transf):
        residual_rel_c2c = []
        for i in range(len(rel_transf)):
            c_res = []
            ref_idx = 0
            for j in range(len(rel_transf[i])):
                curr_transf = rel_transf[i][j]
                prev_idx = max(0, j-1)
                prev_inv_transf = np.linalg.inv(rel_transf[i][prev_idx])
                residual_transf = curr_transf @ prev_inv_transf
                c_res.append(residual_transf)
            c_res = np.array(c_res, dtype=rel_transf[i].dtype)
            residual_rel_c2c.append(c_res)
        return residual_rel_c2c
    
    def compute_camera_transformation(self, rel_transf, rel_indices):
        # compute camera translation and rotation w.r.t. reference frame
        self.cam_transl, self.cam_rot = [], []
        self.cam_transf_mask = []
        for i in range(len(rel_indices)):
            c_cam_transl, c_cam_rot = [], []
            c_transf_mask = np.ones(len(rel_transf[rel_indices[i]]))
            for j in range(len(rel_transf[rel_indices[i]])):
                curr_transf = rel_transf[rel_indices[i]][j]
                curr_inv_transf = np.linalg.inv(curr_transf)
                if np.isnan(curr_inv_transf).any():
                    c_transf_mask[j] = 0
                    continue
                c_cam_transl.append(curr_inv_transf[:3, 3])
                c_cam_rot.append(curr_inv_transf[:2,:3].reshape(-1)) # 6D rotation representation
            c_cam_transl, c_cam_rot = np.array(c_cam_transl), np.array(c_cam_rot)
            self.cam_transl.append(c_cam_transl)
            self.cam_rot.append(c_cam_rot)
            self.cam_transf_mask.append(c_transf_mask)
    
    def compute_contact_translation(self, rel_transf, rel_indices):
        # compute relative translation b/w contact points w.r.t. reference contact point
        self.contact_transl_rel = []
        self.contact_transl_rel_mask = []
        for i in range(len(self.contact_centroids)):
            ref_idx = 0
            c_contact = self.contact_centroids[i]
            c_contact_transl_rel = np.zeros((len(c_contact), 3))
            c_contact_transl_rel_mask = np.ones(len(c_contact)) * self.contact_centroids_mask[i][ref_idx]
            for j in range(len(c_contact)):
                # transform contact centroid to reference frame
                c_contact_transl_rel[j] = (np.linalg.inv(rel_transf[rel_indices[i]][j]) @ np.concatenate([c_contact[j], [1]]))[:3]
                c_contact_transl_rel[j] -= c_contact[ref_idx]
                if np.isnan(c_contact_transl_rel[j]).any():
                    c_contact_transl_rel_mask[j] = 0
            
            self.contact_transl_rel.append(c_contact_transl_rel)
            self.contact_transl_rel_mask.append(c_contact_transl_rel_mask)

    def compute_contact_volumes(self):
        # create 3D grid for contact volumes
        # with gaussian kernel at each contact point for each frame
        # contact volume is a 3D tensor of size (T, D, D, D)
        # where T is the number of frames, D is the resolution of the grid
        grid_size = self.opt.contact_grid
        if 'holo' in self.opt.dataset_name:
            save_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_settings_data_precomputed')
        elif 'arctic' in self.opt.dataset_name:
            save_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/arctic_data_precomputed')
        if self.opt.setting is not None:
            save_file = pjoin(save_dir, f'{self.opt.dataset_name}_motion_{self.opt.setting}_{self.split}_{self.opt.window_size:02d}.pkl')
        else:
            save_file = pjoin(save_dir, f'{self.opt.dataset_name}_motion_{self.split}_{self.opt.window_size:02d}.pkl')
        if os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                save_dict = pickle.load(f)
            for k, v in save_dict.items():
                setattr(self, k, v)
            print (f'Loaded contact volumes for {len(self.contact_volumes)} motions')
        else:
            self.contact_volumes = []
            self.contact_masks = []
            self.grid_volumes = []
            self.grid_masks = []
            self.contact_centroids = []
            self.contact_centroids_mask = []

            range_dir = os.path.join(os.environ['ROOT_DIR'], 'data/ranges')
            range_file = pjoin(range_dir, f'{self.opt.dataset_name}_motion_{self.opt.setting}_{self.opt.window_size:02d}.pkl')

            if 'train' in self.split:
                # compute a fixed grid range from train split and save it
                x_range, y_range, z_range = (1e9, -1e9), (1e9, -1e9), (1e9, -1e9)
                for idx, contact in enumerate(self.contacts):
                    x_r, y_r, z_r = self.compute_xyz_range(contact)
                    if x_r is None or y_r is None or z_r is None:
                        continue
                    x_range = (min(x_range[0], x_r[0]), max(x_range[1], x_r[1]))
                    y_range = (min(y_range[0], y_r[0]), max(y_range[1], y_r[1]))
                    z_range = (min(z_range[0], z_r[0]), max(z_range[1], z_r[1]))
                
                # sanity check although this should not happen, that would mean no valid sequence in the entire train split
                if x_range[0] == 1e9 or y_range[0] == 1e9 or z_range[0] == 1e9:
                    x_range, y_range, z_range = None, None, None
                if x_range[1] == -1e9 or y_range[1] == -1e9 or z_range[1] == -1e9:
                    x_range, y_range, z_range = None, None, None
                # save the range
                range_dict = {'x_range': x_range, 'y_range': y_range, 'z_range': z_range}
                os.makedirs(range_dir, exist_ok=True)
                with open(range_file, 'wb') as f:
                    pickle.dump(range_dict, f)
            else:
                # load the range for val and test splits
                with open(range_file, 'rb') as f:
                    range_dict = pickle.load(f)
                x_range = range_dict['x_range']
                y_range = range_dict['y_range']
                z_range = range_dict['z_range']

            for idx, contact in enumerate(tqdm(self.contacts)):
                if x_range is None or y_range is None or z_range is None:
                    # no contact points in the sequence
                    self.contact_volumes.append(np.zeros((len(contact), grid_size, grid_size, grid_size)))
                    self.grid_volumes.append(np.zeros((grid_size, grid_size, grid_size, 3)))
                    self.contact_masks.append(np.zeros(len(contact)))
                    self.grid_masks.append(0)
                    self.contact_centroids.append(np.ones((len(contact), 3)) * -1)
                    self.contact_centroids_mask.append(np.zeros(len(contact)))
                    continue
                
                # create a 3D grid normalized to [0, 1] in x_range, y_range, z_range
                x_grid = np.linspace(x_range[0], x_range[1], grid_size)
                y_grid = np.linspace(y_range[0], y_range[1], grid_size)
                z_grid = np.linspace(z_range[0], z_range[1], grid_size)
                x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid)
                grid = np.stack([x_grid, y_grid, z_grid], axis=-1)
                self.grid_volumes.append(grid)
                self.grid_masks.append(1)
                
                near, far = np.array([x_range[0], y_range[0], z_range[0]]), np.array([x_range[1], y_range[1], z_range[1]])
                # compute contact volume
                contact_volume = np.zeros((len(contact), grid_size, grid_size, grid_size))
                contact_mask = np.ones(len(contact))
                contact_centroids = np.ones((len(contact), 3)) # this should not matter since these are masked out
                contact_centroids_mask = np.zeros(len(contact))
                for i, frame in enumerate(contact):
                    if np.isnan(frame).any():
                        contact_mask[i] = 0
                        continue
                    if len(frame) == 0:
                        contact_mask[i] = 0
                        continue
                    contact_centroids[i] = np.mean(frame, axis=0)
                    if not np.isnan(contact_centroids[i]).any():
                        contact_centroids_mask[i] = 1
                    for j, contact_point in enumerate(frame):
                        # normalize contact point to x_range, y_range, z_range
                        contact_norm = (contact_point - near) / (far - near)
                        contact_volume[i] = self.compute_gaussian_heatmap(contact_norm, contact_volume[i])
                
                self.contact_volumes.append(contact_volume)
                self.contact_masks.append(contact_mask)
                self.contact_centroids.append(contact_centroids)
                self.contact_centroids_mask.append(contact_centroids_mask)

                # if self.opt.debug and idx > 5:
                #     break

            # save the computed contact volumes
            # save all attributes in a pickle file
            rel_keys = ['contact_volumes', 'contact_masks', 'grid_volumes', 'grid_masks', 'contact_centroids', 'contact_centroids_mask']
            save_dict = {}
            for k in rel_keys:
                if hasattr(self, k):
                    save_dict[k] = getattr(self, k)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_file, 'wb') as f:
                pickle.dump(save_dict, f)
            print (f'Saved contact volumes for {len(self.contact_volumes)} motions')
    
    def compute_xyz_range(self, curr_contact):
        # curr_contact is list of Nx3 numpy arrays with nans
        # compute the range of x, y, z values for entire list
        # return the range as a tuple
        clean_cnt = self.remove_nans(curr_contact)
        if len(clean_cnt) == 0:
            return None, None, None
        all_contacts = np.concatenate(clean_cnt)
        if len(all_contacts) == 0: # verify this
            return None, None, None
        x_range = (np.nanmin(all_contacts[:, 0]), np.nanmax(all_contacts[:, 0]))
        y_range = (np.nanmin(all_contacts[:, 1]), np.nanmax(all_contacts[:, 1]))
        z_range = (np.nanmin(all_contacts[:, 2]), np.nanmax(all_contacts[:, 2]))
        return x_range, y_range, z_range
    
    def remove_nans(self, array_list):
        # remove nans from array list
        # array_list is a list of numpy arrays
        # return list of arrays with nans removed
        out = []
        for array in array_list:
            if np.isnan(array).any():
                continue
            out.append(array)
        return out
    
    def compute_gaussian_heatmap(self, point, volume, sigma=1.0):
        """
        Compute a 3D Gaussian heatmap around a point within a given volume.

        Args:
            point (np.ndarray): The point around which to create the Gaussian heatmap (normalized to [0, 1]).
            volume (np.ndarray): The 3D volume grid.
            sigma (float): The standard deviation of the Gaussian.

        Returns:
            np.ndarray: The volume with the Gaussian heatmap added.
        """
        grid_size = volume.shape[0]
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = self.gaussian_kernel(kernel_size, sigma)
        
        # Compute the center of the kernel in the volume
        center = (point * (grid_size - 1)).astype(int)
        
        # Define the ranges for the kernel placement
        x_start = max(center[0] - kernel_size // 2, 0)
        x_end = min(center[0] + kernel_size // 2 + 1, grid_size)
        y_start = max(center[1] - kernel_size // 2, 0)
        y_end = min(center[1] + kernel_size // 2 + 1, grid_size)
        z_start = max(center[2] - kernel_size // 2, 0)
        z_end = min(center[2] + kernel_size // 2 + 1, grid_size)
        
        # Define the ranges for the kernel itself
        kx_start = max(kernel_size // 2 - center[0], 0)
        kx_end = min(kernel_size // 2 + (grid_size - center[0]), kernel_size)
        ky_start = max(kernel_size // 2 - center[1], 0)
        ky_end = min(kernel_size // 2 + (grid_size - center[1]), kernel_size)
        kz_start = max(kernel_size // 2 - center[2], 0)
        kz_end = min(kernel_size // 2 + (grid_size - center[2]), kernel_size)
        
        # Add the Gaussian kernel to the volume
        volume[x_start:x_end, y_start:y_end, z_start:z_end] += kernel[kx_start:kx_end, ky_start:ky_end, kz_start:kz_end]
        
        return volume

    def gaussian_kernel(self, size=5, sigma=1.0):
        """
        Generate a 3D Gaussian kernel.

        Args:
            size (int): The size of the kernel (size x size x size).
            sigma (float): The standard deviation of the Gaussian.

        Returns:
            np.ndarray: 3D Gaussian kernel.
        """
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy, zz = np.meshgrid(ax, ax, ax)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        return kernel


class CodebookIndicesData(HoloMotion):
    def __init__(self, opt, split='train'):
        super(CodebookIndicesData, self).__init__(opt, split=split)
        self.opt = opt

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        name = data['name']
        start = data['start']
        end = data['end']
        ranges = data['range']
        motion_id = data['motion_id']

        if self.opt.load_indices is not None:
            curr_idx = self.indices[name][f'{start}_{end}']['indices'] # (seq_len * num_quantizers)
        text_feats = data['text_feats'][0] # (seq_len, text_dim), all are same
        video_feats = data['video_feats'][0] # (seq_len, video_dim), only need the first image
        contact_map = data['contact'][0] # only need the first one
        contact_mask = data['contact_mask'][0] # only need the first one
        grid_map = data['grid'] # all timesteps are same
        grid_mask = data['grid_mask'] # all timesteps are same
        contact_point = self.contact_centroids[motion_id][start]

        if 'contact_ref_mask' in data:
            rel_mask = data['contact_mask'][0] * data['contact_ref_mask'][0] * data['joints_ref_mask'][0]
        else:
            rel_mask = data['contact_mask'][0] #* data['joints_ref_mask'][0]

        curr_dict = {
            'name': name,
            'start': start,
            'end': end,
            'range': ranges,
            'motion_id': motion_id,
            'text_feats': text_feats,
            'video_feats': video_feats,
            'contact': contact_map,
            'contact_mask': contact_mask,
            'grid': grid_map,
            'grid_mask': grid_mask,
            'contact_point': contact_point,
            # 'code_idx': curr_idx,
            'rel_mask': rel_mask
        }
        if self.opt.load_indices is not None:
            curr_dict['code_idx'] = curr_idx

        return curr_dict


class ArcticMotion(HoloMotion):
    def __init__(self, opt, split='train'):
        if opt is not None:
            self.opt = opt
        else:
            self.opt = argparse.Namespace()
            self.opt.window_size = 10
            self.opt.joints_num = 21
        self.split = split

        data_mode = 'train'
        if split != 'train':
            data_mode = 'val'
        
        arctic_contact_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/arctic_contact_dir')
        file = f'{arctic_contact_dir}/arctic_{data_mode}_r.pkl'
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # run the check once when contact data is preprocessed again
        # verify = self.sanity_check(data)
        # if not verify:
        #     raise ValueError('Data is not consistent with the transformation matrix')
        
        if 'joints' in self.opt.motion_type:
            self.motion = data[self.opt.motion_type]
            self.motion = [m.reshape(len(m), -1) for m in self.motion]
            self.mean = np.mean(np.concatenate(self.motion), axis=0)
            self.std = np.std(np.concatenate(self.motion), axis=0)
        elif 'mano' in self.opt.motion_type:
            self.thetas = data['thetas']
            self.thetas = [m.reshape(len(m), -1) for m in self.thetas]
            self.thetas_mean = np.zeros_like(self.thetas[0][0])
            self.thetas_std = np.ones_like(self.thetas[0][0])
            self.betas = data['betas']
            self.betas = [m.reshape(len(m), -1) for m in self.betas]
            self.betas_mean = np.zeros_like(self.betas[0][0])
            self.betas_std = np.ones_like(self.betas[0][0])
            self.cam_t = data['cam_t']
            self.cam_t = [m.reshape(len(m), -1) for m in self.cam_t]
            self.cam_t_mean = np.mean(np.concatenate(self.cam_t), axis=0) # these are zerod out later, consisteny with Holo processing
            self.cam_t_std = np.std(np.concatenate(self.cam_t), axis=0)
            assert len(self.thetas) == len(self.betas) == len(self.cam_t)
            # concatenate all
            self.mano = [np.concatenate([self.thetas[i], self.betas[i], self.cam_t[i]], axis=-1) for i in range(len(self.thetas))]
            self.mean = np.concatenate([self.thetas_mean, self.betas_mean, self.cam_t_mean], axis=-1)
            self.std = np.concatenate([self.thetas_std, self.betas_std, self.cam_t_std], axis=-1)
        else:
            raise ValueError('Unknown motion type')
        
        if self.opt.video_feats is not None:
            feats_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/arctic_video_feats')
            feats_file = f'{feats_dir}/arctic_feats_{data_mode}.pkl'
            feats = pickle.load(open(feats_file, 'rb'))

            if self.opt.video_feats == 2048:
                feat_prefix = 'conv_'
            elif self.opt.video_feats == 768:
                feat_prefix = 'tf_'

            vid_feats = []
            for idx in range(len(data['names'])):
                name, (st, end) = data['names'][idx], data['ranges'][idx]
                vid_feats.append(feats[name][(st, end)][f'{feat_prefix}feat'])
            assert len(vid_feats) == len(data['names'])
            self.feats = []

        if self.opt.text_feats:
            self.text_feats = []
            if self.opt.video_feats is None:
                feats_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/arctic_video_feats')
                feats_file = f'{feats_dir}/arctic_feats_{data_mode}.pkl'
                feats = pickle.load(open(feats_file, 'rb'))
            
            self.texts = []
            for idx in range(len(data['names'])):
                name, (st, end) = data['names'][idx], data['ranges'][idx]
                self.texts.append(feats[name][(st, end)]['text_feat'])
            
            assert len(self.texts) == len(data['names'])

        if self.opt.contact_grid is not None:
            self.contacts = []

        if self.opt.contact_map:
            self.contact_maps = []

        data_joints_ref = data['joints_ref']
        data_joints_ref = [m.reshape(len(m), -1) for m in data_joints_ref]
        self.joints_ref = []

        self.data = [] # list of sequences
        self.lengths = [] # length of each sequence
        self.names = [] # video name
        self.ranges = [] # start, end of each sequence in the video
        self.tasks = [] # task for each sequence

        relevant_list = self.motion if 'joints' in self.opt.motion_type else self.mano
        relevant_indices = []
        for j, curr_motion in enumerate(relevant_list):
            if curr_motion.shape[0] < self.opt.window_size:
                continue
            
            # don't need aggressive filtering here since arctic is clean data
            # should be fine even without filtering (haven't verified this yet)
            # filter based on self.cam_t_mean and self.cam_t_std
            if 'mano' in self.opt.motion_type:
                if np.any(np.abs(self.cam_t[j] - self.cam_t_mean[None]) > 2*self.cam_t_std[None]):
                    continue

            relevant_indices.append(j)
            self.lengths.append(curr_motion.shape[0] - self.opt.window_size)
            self.data.append(curr_motion)
            self.names.append(data['names'][j])
            self.ranges.append(data['ranges'][j])
            self.tasks.append(data['tasks'][j])
            if self.opt.video_feats is not None:
                self.feats.append(vid_feats[j])
            if self.opt.text_feats:
                self.text_feats.append(self.texts[j])
            if self.opt.contact_grid is not None:
                self.contacts.append(data['contacts'][j])
            self.joints_ref.append(data_joints_ref[j])
            if self.opt.contact_map:
                self.contact_maps.append(data['contact_maps'][j])

        self.rel_c2c = [data['rel_c2c'][i] for i in relevant_indices]

        if opt.contact_grid is not None:
            self.compute_contact_volumes()
        
        # set self.cam_transf_mask to all ones
        self.cam_transf_mask = [np.ones(len(self.data[i])) for i in range(len(self.data))]

        self.reset_mean_std()

        # set joint_ref_mask to all ones
        self.joints_ref_mask = [np.ones(len(self.data[i])) for i in range(len(self.data))]
        self.update_joints_ref_mask()

        if self.opt.pred_cam:
            # compute camera translation and rotation w.r.t. reference frame
            self.compute_camera_transformation(data['rel_c2c'], relevant_indices)

            # append camera transformation matrix to the end of the self.data
            self.data = [np.concatenate([self.data[i], self.cam_rot[i]], axis=-1) for i in range(len(self.data))] # (T, 58+3+3+3) -> (T, 58+3+3+3+6)
            self.mean = np.concatenate([self.mean, np.zeros(6)], axis=-1)
            self.std = np.concatenate([self.std, np.ones(6)], axis=-1)

            self.data = [np.concatenate([self.data[i], self.cam_transl[i]], axis=-1) for i in range(len(self.data))] # (T, 58+3+3+3+6) -> (T, 58+3+3+3+6+3)
            self.mean = np.concatenate([self.mean, np.zeros(3)], axis=-1)
            self.std = np.concatenate([self.std, np.ones(3)], axis=-1)
        
        if self.opt.decoder_only and self.opt.load_indices is not None:
            indices_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.load_indices, 'model')
            prefix = ''
            check = 'finest'
            if self.opt.transfer_from is not None:
                transfer_dataset, transfer_model = self.opt.transfer_from.split('/')
                prefix = f'transfer_{transfer_dataset}_{transfer_model}_'
                if 'prior' in transfer_dataset:
                    check = 'prior'
            indices = glob.glob(f'{indices_dir}/{prefix}*{check}*{self.split}*.pkl')
            if self.opt.transfer_from is None:
                # remove any entry with 'transfer' in it
                indices = [i for i in indices if 'transfer' not in i]
            indices.sort()
            if len(indices) == 0:
                indices = glob.glob(f'{indices_dir}/*{self.split}*.pkl')
                indices.sort()
                if len(indices) == 0:
                    print (f'No indices file found in {indices_dir}, Exiting...')
                    sys.exit()
            indices_file = indices[-1]
            self.indices = pickle.load(open(indices_file, 'rb'))

        self.mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def __getitem__(self, item):
        data = super().__getitem__(item)
        
        return data


class ArcticCodebookIndicesData(ArcticMotion):
    def __init__(self, opt, split='train'):
        super(ArcticCodebookIndicesData, self).__init__(opt, split=split)
        self.opt = opt

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        name = data['name']
        start = data['start']
        end = data['end']
        ranges = data['range']
        motion_id = data['motion_id']

        if self.opt.load_indices is not None and (self.opt.transfer_from is None or 'prior' not in self.opt.transfer_from):
            curr_idx = self.indices[name][f'{start}_{end}']['indices'] # (seq_len * num_quantizers)
        text_feats = data['text_feats'][0] # (seq_len, text_dim), all are same
        video_feats = data['video_feats'][0] # (seq_len, video_dim), only need the first image
        contact_map = data['contact'][0] # only need the first one
        contact_mask = data['contact_mask'][0] # only need the first one
        grid_map = data['grid'] # all timesteps are same
        grid_mask = data['grid_mask'] # all timesteps are same
        contact_point = self.contact_centroids[motion_id][start]

        if 'contact_ref_mask' in data:
            rel_mask = data['contact_mask'][0] * data['contact_ref_mask'][0] * data['joints_ref_mask'][0]
        else:
            rel_mask = data['contact_mask'][0]

        curr_dict = {
            'name': name,
            'start': start,
            'end': end,
            'range': ranges,
            'motion_id': motion_id,
            'text_feats': text_feats,
            'video_feats': video_feats,
            'contact': contact_map,
            'contact_mask': contact_mask,
            'grid': grid_map,
            'grid_mask': grid_mask,
            'contact_point': contact_point,
            # 'code_idx': curr_idx,
            'rel_mask': rel_mask
        }
        if self.opt.load_indices is not None and (self.opt.transfer_from is None or 'prior' not in self.opt.transfer_from):
            curr_dict['code_idx'] = curr_idx

        return curr_dict