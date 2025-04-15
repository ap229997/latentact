import os
import pickle
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import *
from torchvision.transforms import Normalize, ToTensor, Compose, Resize, CenterCrop

import clip

parser = argparse.ArgumentParser()
parser.add_argument('--start_num', type=int, default=None)
parser.add_argument('--end_num', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_threads', type=int, default=8)
parser.add_argument('--use_inpaint', action='store_true')
parser.add_argument('--save_dir', type=str, default='logs/debug_feats')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

torch.set_num_threads(args.num_threads)

if args.start_num is None:
    args.start_num = 0
if args.end_num is None:
    args.end_num = 100

def check_range(curr_range, target_range):
    st, end = curr_range
    target_st, target_end = target_range
    # Check if both st and end are within the bounds of target_range
    return target_st <= st <= target_end and target_st <= end <= target_end
    

def load_contacts(contact_dir):
    files = sorted(glob.glob(f'{contact_dir}/contact_*_fix.pkl'))
    contact_data = {}
    # combine dicts from all pkl files
    for file in tqdm(files):
        strs = file.split('/')[-1].split('_')
        st, end = int(strs[1]), int(strs[2])
        if not check_range((st, end), (args.start_num, args.end_num)):
            continue
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

contact_dir = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_contact_dir')
data = load_contacts(contact_dir)

encoder = resnet50(pretrained=True).cuda()
# encoder.avgpool = torch.nn.Identity()
encoder.fc = torch.nn.Identity()

vit_encoder = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
vit_encoder.head = torch.nn.Identity()
vit_encoder.to('cuda')

clip_encoder, _ = clip.load("ViT-B/32", device='cuda')

holo_path = os.environ['HOLO_PATH']

# inpainting done using Affordance Diffusion (https://github.com/NVlabs/affordance_diffusion)
# using hand masks extracted earlier, set the path here after inpainting is done
inpaint_path = os.environ.get('INPAINT_PATH', None)
use_inpaint = args.use_inpaint

# resize images to 224x224 (maintain aspect ratio with zero padding) and normalize to imagenet mean and std
# padding is done in the dataset class below
normalize_img = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, name, ranges, text) -> None:
        super().__init__()
        self.name = name
        self.st, self.end = ranges
        self.text = text
        self.use_inpaint = use_inpaint
        if use_inpaint:
            self.inpaint_dir = os.path.join(inpaint_path, name)
        self.img_dir = os.path.join(holo_path, name, 'Export_py/Video/images_jpg')

    def __len__(self):
        return self.end - self.st + 1

    def __getitem__(self, idx):
        if self.use_inpaint:
            img_dir = self.inpaint_dir
        else:
            img_dir = self.img_dir
        img_path = os.path.join(img_dir, f'{self.st+idx:06d}.jpg')
        if not os.path.exists(img_path):
            img_dir = self.img_dir
            img_path = os.path.join(img_dir, f'{self.st+idx:06d}.jpg')
        img = Image.open(img_path)
        max_dim = max(img.size)
        pad = (max_dim - img.size[0], max_dim - img.size[1])
        img = np.array(img)
        # pad image to make it square, keep the image in the center
        img = np.pad(img, ((pad[1]//2, pad[1] - pad[1]//2), (pad[0]//2, pad[0] - pad[0]//2), (0, 0)), mode='constant')
        img = Image.fromarray(img)
        norm_img = normalize_img(img)
        return norm_img


feat_dict = {}
for idx in tqdm(range(len(data['names']))):
    name, (st, end), text = data['names'][idx], data['ranges'][idx], data['tasks'][idx]

    curr_dataset = ImageDataset(name, (st, end), text)
    dataloader = DataLoader(curr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    curr_dict = {'name': name, 'range': (st, end)}
    c_feat, t_feat = [], []
    for batch in dataloader:
        all_imgs = batch
        with torch.no_grad():
            all_imgs = all_imgs.to('cuda')
            c_feat.append(encoder(all_imgs).cpu().detach().numpy())
            t_feat.append(vit_encoder(all_imgs).cpu().detach().numpy())
        
    feat = np.concatenate(c_feat, axis=0)
    curr_dict.update({'conv_feat': feat})

    tf_feat = np.concatenate(t_feat, axis=0)
    curr_dict.update({'tf_feat': tf_feat})
    
    if args.debug:
        break

    # compute clip feature of text
    text_token = clip.tokenize(text).cuda()
    text_feat = clip_encoder.encode_text(text_token)
    # repeat text feat for all images
    text_feat = text_feat.repeat(len(curr_dataset), 1).cpu().detach().numpy()
    curr_dict.update({'text_feat': text_feat})

    if name not in feat_dict:
        feat_dict[name] = {}
    feat_dict[name][(st,end)] = curr_dict
    # break

suffix = ''
if use_inpaint:
    suffix = '_inpaint'
# save feat_dict for each video separately as a pickle file
save_dir = f'{args.save_dir}/holo_contact_feats{suffix}'
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, f'holo_feats_{args.start_num:04d}_{args.end_num:04d}.pkl')
with open(save_file, 'wb') as f:
    pickle.dump(feat_dict, f)
