import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import tqdm
import sys
import json
from ffprobe import FFProbe

from common.holo_utils import get_video_subset, sample_clips
from common.generic_utils import reset_all_seeds


def get_framerate(filename):
    metadata=FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            #print("stream",stream.__dir__())
            print('Stream contains {} frames.'.format(stream.framerate))
            framerate = float(stream.framerate)
            frame_num = int(stream.frames())
            
    return framerate, frame_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=None, help='Directory of untar data')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample clips from the video')
    parser.add_argument('--task_type', nargs='+', default=[''], help='Task type to process')
    parser.add_argument('--extract_frames', action='store_true', help='Extract frames from video')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    parser.add_argument('--start_num', type=int, default=None, help='start video index')
    parser.add_argument('--end_num', type=int, default=None, help='end video index')
    args = parser.parse_args()

    reset_all_seeds(seed=42) # fix seed for reproducibility

    if args.folder_path is None:
        args.folder_path = os.environ['HOLO_PATH']

    if args.video_name is not None:
        video_names = [args.video_name]
    else:
        video_names = os.listdir(args.folder_path)
        video_names.sort()

    if not args.all:
        # load action annotations and language descriptions
        annot_path = os.path.join(os.path.dirname(os.path.abspath(args.folder_path)), 'data-annotation-trainval-v1_1.json')
        assert os.path.exists(annot_path), f'Annotation file {annot_path} does not exist.'
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
    
    relevant_videos = 0
    skipped_videos = 0
    for video_name in tqdm.tqdm(video_names):
        if args.debug:
            video_name = 'z095-july-11-22-knarrevik_disassemble'
        base_path = os.path.join(args.folder_path, video_name, "Export_py")

        if not os.path.exists(base_path):
            # Exit if the path does not exist
            print ('{} does not exist'.format(base_path))
            skipped_videos += 1
            continue
            
        mpeg_img_path = os.path.join(base_path,"Video","images_jpg")
        if not os.path.exists(os.path.join(base_path,"Video_pitchshift.mp4")):
            print ('{} does not exist, Skipping ...'.format(os.path.join(base_path,"Video_pitchshift.mp4")))
            skipped_videos += 1
            continue

        relevant_videos += 1
        
        if args.extract_frames is not None and not os.path.exists(mpeg_img_path):
            os.chdir(os.path.join(base_path,"Video"))
            os.mkdir(mpeg_img_path)
            print ('Extracting frames from video {}'.format(video_name))
            os.system("ffmpeg -i ../Video_pitchshift.mp4 -q:v 1 -start_number 0 images_jpg/%06d.jpg")

        if args.debug:
            break

    print (f'Found {relevant_videos} relevant videos and {skipped_videos} missing videos.')


if __name__ == '__main__':
    main()