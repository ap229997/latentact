# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import pickle
import argparse

import cv2
import torch
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sam2.build_sam import build_sam2_camera_predictor

from common.generic_utils import reset_all_seeds
from common.holo_utils import get_video_subset, sample_clips


def show_mask(mask, ax, obj_id=None, random_color=False):
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_bbox(bbox, ax, marker_size=200):
    tl, br = bbox[0], bbox[1]
    w, h = (br - tl)[0], (br - tl)[1]
    x, y = tl[0], tl[1]
    print(x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=None, edgecolor="blue", linewidth=2))


def mask2box(mask):
    # get the bounding box of the mask
    obj_box = cv2.boundingRect((mask*255).astype(np.uint8))
    if isinstance(obj_box, tuple):
        obj_box = list(obj_box)
    elif isinstance(obj_box, np.ndarray):
        obj_box = obj_box.tolist()
    # convert to [x0, y0, x1, y1]
    obj_box[2] = obj_box[0] + obj_box[2]
    obj_box[3] = obj_box[1] + obj_box[3]
    return obj_box


# refer https://github.com/Gy920/segment-anything-2-real-time for details
def main(video_name, start, end, points, predictor, args=None):
    video_dir = os.path.join(args.holo_path, video_name, 'Export_py/Video/images_jpg')
    # load frames
    frame_names = [os.path.join(video_dir, f"{i:06d}.jpg") for i in range(start, end+1)]
    frame_names.sort()
    frames = [cv2.imread(os.path.join(video_dir, p)) for p in frame_names]

    if len(frames) == 0:
        print (f'No frames found for {video_name} from {start} to {end}.')
        return None

    # process first frame
    frame = frames[0]
    predictor.load_first_frame(frame)

    ann_frame_idx = 0  # the frame index we interact with, always start with the first frame
    ann_obj_id = (1)  # give a unique id to each object we interact with (it can be any integers)

    # using point prompt
    labels = np.ones(len(points), dtype=np.int8)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(12, 8))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(frame)

    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    first_mask = (out_mask_logits[0] > 0.0).cpu().numpy() # (1, H, W) np.bool

    obj_masks = {}
    obj_box = mask2box(first_mask[0])
    obj_masks[start+ann_frame_idx] = {'mask': Image.fromarray(first_mask[0]), 'box': obj_box}

    save_seg = False
    if args.save_mask:
        # randomly assign True with probability p
        save_name = f"{video_name}_{start:06d}_{end:06d}"
        save_p = 0.02
        save_seg = np.random.rand() < save_p
        if save_seg:
            seg_dir = os.path.join(args.out_folder, save_name)
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)

    if save_seg:
        plt.close()
        plt.figure(figsize=(12, 8))
        plt.title(f"frame {start+ann_frame_idx:06d}")
        plt.imshow(frame)
        show_points(points, labels, plt.gca())
        show_mask(first_mask, plt.gca(), obj_id=out_obj_ids[0])
        plt.savefig(os.path.join(seg_dir, f"frame_{start+ann_frame_idx:06d}.jpg"))

    vis_gap = 4
    for ann_frame_idx in range(1, len(frames)):
        frame = frames[ann_frame_idx]

        try:
            out_obj_ids, out_mask_logits = predictor.track(frame)
        except:
            print (f'Frame {ann_frame_idx} failed for {video_name}.')
            continue

        curr_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        obj_box = mask2box(curr_mask[0])
        obj_masks[start+ann_frame_idx] = {'mask': Image.fromarray(curr_mask[0]), 'box': obj_box}

        if ann_frame_idx % vis_gap == 0 and save_seg:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.close()
            plt.figure(figsize=(12, 8))
            plt.title(f"frame {start+ann_frame_idx:06d}")
            plt.imshow(frame)
            show_mask(curr_mask, plt.gca(), obj_id=out_obj_ids[0])
            plt.savefig(os.path.join(seg_dir, f"frame_{start+ann_frame_idx:06d}.jpg"))

    # clear all memory
    del frames
    plt.close('all')

    return obj_masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--holo_path', type=str, default=None, help='path to holoassist data')
    parser.add_argument('--out_folder', type=str, help='output folder')
    parser.add_argument('--task_type', type=str, nargs='+', default=[''], help='task type to filter videos')
    parser.add_argument('--subsample', type=int, help='subsample clips')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--points_file', type=str, default=None, help='json file where clicked points are stored')
    parser.add_argument('--save_mask', action='store_true', help='save masks in pkl format')
    parser.add_argument('--start_num', type=int, default=None, help='start video index')
    parser.add_argument('--end_num', type=int, default=None, help='end video index')
    args = parser.parse_args()

    reset_all_seeds(seed=42)

    if args.holo_path is None:
        args.holo_path = os.environ['HOLO_PATH']
    assert os.path.exists(args.holo_path), f'{args.holo_path} does not exist.'

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root_dir = os.environ['ROOT_DIR']
    sam2_checkpoint = os.path.join(root_dir, "data_engine/segment-anything-2-real-time/checkpoints/sam2_hiera_large.pt")
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    if args.points_file is None:
        ########## test on a few vidoes ##########
        # video_name = 'z009-june-15-22-nespresso' # 'z185-sep-09-22-nespresso' # 'z009-june-15-22-nespresso'
        # start, end = 1524, 1544 # 1350, 1300 # 277, 324 # 1524, 1564
        # input_points = np.array([[370, 340], [450, 330]], dtype=np.float32)
        # first_mask_id = 0

        # z127-aug-10-22-nespresso 296 370 [[[500, 230]]]
        video_name = 'R012-7July-Nespresso' # 'z127-aug-10-22-nespresso' # 'R012-7July-Nespresso'
        start, end = 165, 178  # 296, 370 # 165, 178
        first_mask_id = 0
        input_points = np.array([[400, 300]], dtype=np.float32)
        
        curr_masks = main(video_name, start, end, input_points, predictor, args=args)
        ########## test on a few vidoes ##########

    else:
        if not os.path.exists(args.points_file):
            args.points_file = os.path.join(root_dir, f'data_engine/segment-anything-2-real-time/{args.points_file}')
        assert os.path.exists(args.points_file), f'{args.points_file} does not exist.'
        
        clicked_points = json.load(open(args.points_file, 'r')) # lists of dicts format: [{'image': 'R100-1Aug-Coffee_004339.jpg', 'x': 422, 'y': 314}]
        # aggregate points for each image as list and store in dict
        points_dict = {}
        for point in clicked_points:
            if point['image'] not in points_dict:
                points_dict[point['image']] = []
            points_dict[point['image']].append([point['x'], point['y']])

            # # visualize points on image
            # vname, iname = point['image'].split('_')
            # img_path = os.path.join(args.holo_path, vname, 'Export_py/Video/images', iname)
            # img = cv2.imread(img_path)
            # cv2.circle(img, (point['x'], point['y']), 5, (0, 255, 0), -1)
            # cv2.imwrite(os.path.join(args.out_folder, point['image']), img)

        annot_path = os.path.join(os.path.dirname(os.path.abspath(args.holo_path)), 'data-annotation-trainval-v1_1.json')
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

        # add points from points_dict to each clip and sample relevant video names
        rel_video_names = {}
        rel_clips = 0
        for video_name, clips in video_names.items():
            for clip in clips:
                start, end, verb, noun, hand_type = clip
                first_image_id = video_name + f'_{start:06d}.jpg'
                if first_image_id in points_dict:
                    if video_name not in rel_video_names:
                        rel_video_names[video_name] = []
                    rel_video_names[video_name].append((start, end, verb, noun, hand_type, points_dict[first_image_id]))
                    rel_clips += 1
        rel_video_names = dict(sorted(rel_video_names.items()))
        print (f'Sampled {rel_clips} clips with clicked points.')

        if args.save_mask:
            if not os.path.exists(args.out_folder):
                os.makedirs(args.out_folder)

        for video_name, clips in tqdm(rel_video_names.items()):
            save_file = os.path.join(args.out_folder, f'{video_name}_obj_masks.pkl')
            if os.path.exists(save_file):
                print (f'{video_name} already processed.')
                continue

            video_obj_masks = {}
            for clip in tqdm(clips):
                start, end, verb, noun, hand_type, points = clip
                clip_obj_masks = main(video_name, start, end, np.array(points), predictor, args=args)
                if clip_obj_masks is not None:
                    video_obj_masks.update(clip_obj_masks)

            if args.save_mask:
                # save masks in pkl format
                with open(save_file, 'wb') as f:
                    pickle.dump(video_obj_masks, f)
                print (f'Saved {video_name} masks in {save_file}.')

            del video_obj_masks
    