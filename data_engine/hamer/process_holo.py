import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
from pathlib import Path
import json

import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

from common.generic_utils import reset_all_seeds
from common.holo_utils import get_video_subset, sample_clips
from common.mano_wrapper import LIGHT_BLUE


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default=None, help='Folder with holo images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--load_boxes', type=str, default=None, help='Path to load hand boxes from file')
    parser.add_argument('--sep_hands', action='store_true', help='Process each hand separately')
    parser.add_argument('--save_hands', action='store_true', help='Save hand mesh')
    parser.add_argument('--use_gt_f', action='store_true', help='Use ground truth focal length')
    parser.add_argument('--all', action='store_true', default=False, help='Process all images in the folder')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample clips from the video')
    parser.add_argument('--task_type', nargs='+', default=[''], help='Task type to process')
    parser.add_argument('--start_num', type=int, default=None, help='start video index')
    parser.add_argument('--end_num', type=int, default=None, help='end video index')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')

    args = parser.parse_args()

    reset_all_seeds(seed=42) # fix seed for reproducibility

    if args.img_folder is None:
        args.img_folder = os.environ['HOLO_PATH']
    assert os.path.exists(args.img_folder), f'Path {args.img_folder} does not exist.'

    if args.load_boxes is None:
        args.load_boxes = os.path.join(os.environ['ROOT_DIR'], 'downloads/holo_hand_bbox')
    # without precomputed boxes, it is too slow when using ViTPose
    assert os.path.exists(args.load_boxes), f'Path {args.load_boxes} does not exist.'

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    if args.all:
        # get video names from input_dir
        video_names = os.listdir(args.img_folder)
        video_names.sort()
    else:
        annot_path = os.path.join(os.path.dirname(os.path.abspath(args.img_folder)), 'data-annotation-trainval-v1_1.json') # present in HoloAssist dataset
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

    if args.all:
        video_names = video_names[args.start_num:args.end_num]
    else:
        video_names = dict(list(video_names.items())[args.start_num:args.end_num])

    for video_name in tqdm(video_names):
        save_name = os.path.join(os.path.dirname(args.out_folder), f'{video_name}_masks_hamer.pkl')
        if not args.debug and os.path.exists(save_name):
            print(f'Found masks for {video_name}, Skipping...')
            continue
        
        if isinstance(video_name, dict):
            video_name = video_name['video_name']
            clips = video_name['clip']

        if args.debug:
            video_name = 'z127-aug-10-22-nespresso' # fix a video
        
        img_dir = f'{args.img_folder}/{video_name}/Export_py/Video/images_jpg'
        if not os.path.exists(img_dir):
            print (f'No images found in {img_dir}')
            video_file= f'{args.img_folder}/{video_name}/Export_py/Video_pitchshift.mp4'
            if not os.path.exists(video_file):
                print(f'No video found in {video_file}, Skipping...')
                continue
            # read video and frames
            cap = cv2.VideoCapture(video_file)
            too_large = False # memory issues in processing large videos
            if not cap.isOpened():
                print(f'Error opening video file {video_file}')
                continue
            curr_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # cv2.imwrite(f'{img_dir}/{frame_id:06d}.png', frame)
                curr_frames.append(frame)
                if len(curr_frames) > 15000:
                    too_large = True
                    break
            cap.release()
            if too_large:
                print(f'Video too large, skipping...')
                continue

        # get images
        if args.all:
            img_ls = glob.glob(f'{img_dir}/*')
            test_img_ls = []
            for img_path in img_ls:
                if os.path.exists(img_path) and img_path.endswith(('.jpg', '.JPEG', '.png', '.PNG')):
                    test_img_ls.append(img_path)
            print(f'Got {len(test_img_ls)} images in from {video_name}.')
            test_img_ls.sort()
        else:
            range_ids = video_names[video_name]
            range_ids = [(x[0], x[1]) for x in range_ids]
            frame_ids = []
            for (start, end) in range_ids:
                frame_ids.extend(list(range(start, end+1)))
            frame_ids = sorted(list(set(frame_ids)))
            # get images from frame_ids
            test_img_ls = [f'{img_dir}/{frame_id:06d}.jpg' for frame_id in frame_ids]
            print(f'Got {len(test_img_ls)} images in from {video_name}.')

            if not os.path.exists(img_dir):
                test_img_ls = [(f'{img_dir}/{frame_id:06d}.jpg', curr_frames[frame_id]) for frame_id in frame_ids]

        print(f'Processing {video_name}, with {len(test_img_ls)} valid images out of {len(os.listdir(img_dir))}')
        
        img_paths = test_img_ls
        
        # load precomputed boxes from Hands23 model
        hand_boxes_file = os.path.join(args.load_boxes, f'{video_name}_hand_bbox.pkl')
        if not os.path.exists(hand_boxes_file):
            print(f'No hand masks found for {video_name}, Skipping...')
            continue

        with open(hand_boxes_file, 'rb') as f:
            hand_boxes = pickle.load(f)
        
        pred_hands = {}
        # Iterate over all images in folder
        for img_path in tqdm(img_paths):

            if not os.path.exists(img_path):
                print(f'Image {img_path} not found, Skipping...')
                continue
            
            if isinstance(img_path, tuple):
                img_path, img_cv2 = img_path
            elif isinstance(img_path, str) or isinstance(img_path, Path):
                img_cv2 = cv2.imread(str(img_path))
            else:
                img_cv2 = img_path[:, :, ::-1] # RGB -> BGR

            bboxes = []
            is_right = []
            idx = int(os.path.basename(img_path).split('.')[0])

            if idx in hand_boxes:
                for hand, v in hand_boxes[idx].items():
                    is_right.append(int(hand == 'right'))
                    bboxes.append(v['box'])

            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes).astype(np.float32)
            right = np.stack(is_right)

            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=args.num_workers)

            all_verts = []
            all_joints = []
            all_cam_t = []
            all_right = []
            all_thetas = []
            all_betas = []
            
            # Setup the renderer for every image and clear memory at the end
            renderer = Renderer(model_cfg, faces=model.mano.faces)
            
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                multiplier = (2*batch['right']-1)
                pred_cam = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                multiplier = (2*batch['right']-1)
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                if args.use_gt_f:
                    scaled_focal_length = 685.88 # ground truth focal length for holoassist
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    is_right = batch['right'][n].cpu().numpy()
                    verts[:,0] = (2*is_right-1)*verts[:,0]
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                    joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    joints[:,0] = (2*is_right-1)*joints[:,0]
                    all_joints.append(joints)

                    curr_global = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy()
                    curr_pose = out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy()
                    curr_betas = out['pred_mano_params']['betas'][n].detach().cpu().numpy()
                    curr_theta = np.concatenate([curr_global, curr_pose], axis=0)
                    all_thetas.append(curr_theta)
                    all_betas.append(curr_betas)
            
            # Render front view
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                
                if args.sep_hands:
                    
                    right_verts = [v for v, r in zip(all_verts, all_right) if r]
                    right_joints = [j for j, r in zip(all_joints, all_right) if r]
                    right_cam_t = [c for c, r in zip(all_cam_t, all_right) if r]
                    right_thetas = [t for t, r in zip(all_thetas, all_right) if r]
                    right_betas = [b for b, r in zip(all_betas, all_right) if r]
                    only_right = [r for r in all_right if r]
                    
                    left_verts = [v for v, r in zip(all_verts, all_right) if not r]
                    left_joints = [j for j, r in zip(all_joints, all_right) if not r]
                    left_cam_t = [c for c, r in zip(all_cam_t, all_right) if not r]
                    left_thetas = [t for t, r in zip(all_thetas, all_right) if not r]
                    left_betas = [b for b, r in zip(all_betas, all_right) if not r]
                    only_left = [r for r in all_right if not r]
                    
                    idx = int(os.path.basename(img_path).split('.')[0])
                    pred_hands[idx] = {}
                    pred_hands[idx]['right'] = {'cam_t': np.array(right_cam_t), 'verts': np.array(right_verts), 'joints': np.array(right_joints),
                                                                 'thetas': np.array(right_thetas), 'betas': np.array(right_betas)}
                    pred_hands[idx]['left'] = {'cam_t': np.array(left_cam_t), 'verts': np.array(left_verts), 'joints': np.array(left_joints),
                                                                 'thetas': np.array(left_thetas), 'betas': np.array(left_betas)}
                
                random_save = False
                # randomly save predictions
                if random.random() >= 0.995:
                    random_save = True
                
                if random_save:
                    # save all hands overlaid on image
                    cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
                    input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                    render_dir = args.out_folder
                    cv2.imwrite(os.path.join(render_dir, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

                del renderer

        if args.save_hands:
            with open(save_name, 'wb') as f:
                pickle.dump(pred_hands, f)
                print (f'Saved hands for {video_name} to {save_name}')
        del pred_hands

        if args.debug:
            break
    
if __name__ == '__main__':
    main()
