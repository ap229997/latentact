import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import tqdm
import json
import pickle
from ffprobe import FFProbe

from common.holo_utils import get_video_subset, sample_clips
from common.generic_utils import reset_all_seeds


axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))


class IndexSearch():
    def __init__(self, time_array):
        self.time_array = time_array
        self.prev = 0
        self.index = 0
        self.len = len(time_array)

    def nearest_neighbor(self, target_time):
        while(target_time > self.time_array[self.index]):
            if self.len - 1 <= self.index:
                return self.index
            self.index += 1
            self.prev = self.time_array[self.index]
        
        if (abs(self.time_array[self.index] - target_time) > abs(self.time_array[self.index-1] - target_time)) and (self.index != 0):
            ret_index = self.index-1
        else:
            ret_index = self.index
        return ret_index


def get_handpose_connectivity():
    # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
    return [
        [0, 1],

        # Thumb
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],

        # Index
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],

        # Middle
        [1, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],

        # Ring
        [1, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        # Pinky
        [1, 21],
        [21, 22],
        [22, 23],
        [23, 24],
        [24, 25]
    ]


def read_pose_txt(img_pose_path):
    img_pose_array = []
    time_ids = []
    with open(img_pose_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = list(map(float, line.split('\t')))
            time_ids.append(int(line.split('\t')[1]))
            # pose = np.array(line_data[1:]).reshape(4, 4)
            # pose = np.dot(axis_transform,pose)
            # line_data[1:] = pose.reshape(-1)
            # print("line_data",line_data)
            # line_data = line.strip().split('\t')
            img_pose_array.append(line_data)
        img_pose_array = np.array(img_pose_array)
    return img_pose_array, time_ids


def read_hand_pose_txt(hand_path):
    #  The format for each entry is: Time, IsGripped, IsPinched, IsTracked, IsActive, {Joint values}, {Joint valid flags}, {Joint tracked flags}
    hand_array = []
    time_ids = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            time_ids.append(int(line.split('\t')[1]))
            line_data = list(map(float, line.split('\t')))
            line_data_reshape = np.reshape(
                line_data[3:-52], (-1, 4, 4))  # For version2: line_data[5:-52]

            line_data_xyz = []
            for line_data_reshape_elem in line_data_reshape:
                # To get translation of the hand joints
                location = np.dot(line_data_reshape_elem,
                                np.array([[0, 0, 0, 1]]).T)
                line_data_xyz.append(location[:3].T[0])

            line_data_xyz = np.array(line_data_xyz).T
            hand = line_data[:4]
            hand.extend(line_data_xyz.reshape(-1))
            hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array, time_ids


def read_intrinsics_txt(img_instrics_path):
    with open(img_instrics_path) as f:
        data = list(map(float, f.read().split('\t')))
        intrinsics = np.array(data[:9]).reshape(3, 3)
        width = data[-2]
        height = data[-1]
    return intrinsics, width, height


def read_gaze_txt(gaze_path):
    gaze_data = []
    time_ids = []
    with open(gaze_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            line_data = list(map(float, line.split('\t')))
            time_ids.append(int(line.split('\t')[1]))
            gaze_data.append(line_data)
        gaze_data = np.array(gaze_data)
    return gaze_data, time_ids


def get_eye_gaze_point(gaze_data, dist):
    origin_homog = gaze_data[2:5]
    direction_homog = gaze_data[5:8]
    direction_homog = direction_homog / np.linalg.norm(direction_homog)
    point = origin_homog + direction_homog * dist

    return point[:3]

def get_framerate(filename):
    metadata=FFProbe(filename)
    for stream in metadata.streams:
        if stream.is_video():
            #print("stream",stream.__dir__())
            # print('Stream contains {} frames.'.format(stream.framerate))
            framerate = float(stream.framerate)
            frame_num = int(stream.frames())
            
    return framerate, frame_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=None, help='Directory of untar data')
    parser.add_argument('--video_name', type=str, default=None, help="Directory sequence")
    parser.add_argument('--frame_num', type=int, default=None, help='Specific number of frame')
    parser.add_argument('--eye', action='store_true', help='Proejct eye gaze on images')
    parser.add_argument('--save_eyeproj', action='store_true', help='Save eyeproj.txt file.')
    parser.add_argument('--save_video', type=str, default=None, help='Save hand_project_mpeg file.')
    parser.add_argument('--save_action', type=str, default=None, help='Save action dict to file.')
    parser.add_argument('--eye_dist', type=float, default=0.5, help='Eyegaze projection dist is 50cm by default')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample clips from the video')
    parser.add_argument('--task_type', nargs='+', default=[''], help='Task type to process')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    parser.add_argument('--start_num', type=int, default=None, help='start video index')
    parser.add_argument('--end_num', type=int, default=None, help='end video index')
    parser.add_argument('--timediff', type=str, default=None, help='Check temporal differences between hand and camera poses and save them')
    args = parser.parse_args()

    reset_all_seeds(seed=42) # fix seed for reproducibility

    if args.folder_path is None:
        args.folder_path = os.environ['HOLO_PATH']
    assert os.path.exists(args.folder_path), f'Folder path {args.folder_path} does not exist.'

    if args.save_video is not None:
        args.save_video = os.path.abspath(args.save_video)
        if not os.path.exists(args.save_video):
            os.makedirs(args.save_video)

    if args.video_name is not None:
        video_names = [args.video_name]
    else:
        video_names = os.listdir(args.folder_path)
        video_names.sort()

    if not args.all:
        # load action annotations and language descriptions
        annot_path = os.path.join(os.path.dirname(args.folder_path), 'data-annotation-trainval-v1_1.json')
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
    
    if args.timediff is not None:
        os.makedirs(args.timediff, exist_ok=True)
    
    for video_name in tqdm.tqdm(video_names):

        if  args.save_action is not None and os.path.exists(os.path.join(args.save_action, f'{video_name}_action.pkl')):
            print (f"{video_name} already processed, Skipping ...")
            continue

        if args.save_video is not None:
            save_path = os.path.join(args.save_video, '{}.mp4'.format(video_name))
            if os.path.exists(save_path):
                continue
        
        base_path = os.path.join(args.folder_path, video_name, "Export_py")

        if not os.path.exists(base_path):
            # Exit if the path does not exist
            print('{} does not exist'.format(base_path))
            continue
            
        mpeg_img_path = os.path.join(base_path,"Video","images")
        hands_path = os.path.join(base_path, 'Hands')
        img_path = os.path.join(base_path, 'Video')

        # Read timing file
        img_sync_timing_path = os.path.join(img_path, 'Pose_sync.txt')
        img_sync_timing_array = []
        with open(img_sync_timing_path) as f:
            lines = f.read().split('\n')
            for line in lines:
                if line == '':  # end of the lines.
                    break
                line_data = int(line.split('\t')[1])
                img_sync_timing_array.append(line_data)
        
        start_time_path = os.path.join(img_path, 'VideoMp4Timing.txt')
        with open(start_time_path) as f:
            lines = f.read().split('\n')
            start_time = int(lines[0])
            end_time = int(lines[1])
        
        if not os.path.exists(os.path.join(base_path,"Video_pitchshift.mp4")):
            print (f'Video {video_name} does not exist, Skipping ...')
            continue
        
        frame_rate, frame_num = get_framerate(os.path.join(base_path,"Video_pitchshift.mp4"))
        img_timing_array = []
        for ii in range(frame_num):
            img_timing_array.append(start_time + int(ii * (1/frame_rate)* 10**7))

        # Read left hand
        left_hand_path = os.path.join(hands_path, 'Left_sync.txt')
        left_hand_array, left_timestamp = read_hand_pose_txt(left_hand_path)
        # Read right hand
        right_hand_path = os.path.join(hands_path, 'Right_sync.txt')
        right_hand_array, right_timestamp = read_hand_pose_txt(right_hand_path)

        # find the nearest hand pose for each frame
        if not (left_timestamp == right_timestamp):
            print (f'Video {video_name} does not have matching hand timestamps, Skipping ...')
            continue
        timestamp_array_hand = np.array(left_timestamp, dtype=np.int64)
        mpeg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(timestamp_array_hand.reshape(-1, 1))
        img2hand_timestamps = []
        for ii , img_timestamp in enumerate(img_timing_array[:]):
            _, mpeg_indices = mpeg_nbrs.kneighbors(np.array(img_timestamp).reshape(-1, 1))
            img2hand_timestamps.append(mpeg_indices[0][0])

        # Read gaze
        if args.eye:
            gaze_path = os.path.join(base_path, "Eyes", "Eyes_sync.txt")
            gaze_array, eye_timestamp = read_gaze_txt(gaze_path)
            # find the nearest gaze for each frame
            timestamp_array_eye = np.array(eye_timestamp, dtype=np.int64)
            mpeg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(timestamp_array_eye.reshape(-1, 1))
            img2eye_timestamps = []
            for ii , img_timestamp in enumerate(img_timing_array[:]):
                _, mpeg_indices = mpeg_nbrs.kneighbors(np.array(img_timestamp).reshape(-1, 1))
                img2eye_timestamps.append(mpeg_indices[0][0])
            gaze_timestamp = gaze_array[:, :2] # check what to do with this
            eyeproj_list = []

            # Project into the image
            projected_path = os.path.join(base_path, "projected_mpeg_img")
            if not os.path.exists(projected_path):
                # Create a new directory because it does not exist
                os.makedirs(projected_path)
        
        # Video setup
        if args.save_video is not None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_out = cv2.VideoWriter(save_path, fourcc, frame_rate, (896, 504), isColor=True)

        num_frames = len(img_timing_array) # len(img_sync_timing_array)
        # Read campose
        img_pose_path = os.path.join(img_path, 'Pose_sync.txt')
        img_pose_array, cam_timestamp = read_pose_txt(img_pose_path)
        assert img_sync_timing_array == cam_timestamp

        # find the nearest cam pose for each frame
        timestamp_array_cam = np.array(cam_timestamp, dtype=np.int64)
        mpeg_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(timestamp_array_cam.reshape(-1, 1))
        img2cam_timestamps = []
        for ii , img_timestamp in enumerate(img_timing_array[:]):
            _, mpeg_indices = mpeg_nbrs.kneighbors(np.array(img_timestamp).reshape(-1, 1))
            img2cam_timestamps.append(mpeg_indices[0][0])

        # Read cam instrics
        img_instrics_path = os.path.join(img_path, 'Intrinsics.txt')
        img_intrinsics, width, height = read_intrinsics_txt(img_instrics_path)

        # check temporal deviations
        left_sync = [left_timestamp[x] for x in img2hand_timestamps]
        cam_sync = [cam_timestamp[x] for x in img2cam_timestamps]
        diff_hand, diff_cam = [], []
        if args.timediff is not None:
            time_diff = {}
        for i in range(num_frames):
            dhand = abs(img_timing_array[i]-left_sync[i])
            dcam = abs(img_timing_array[i]-cam_sync[i])
            diff_hand.append(dhand)
            diff_cam.append(dcam)
            if args.timediff is not None:
                time_diff[i] = {'hand': dhand, 'cam': dcam}
        diff_hand = np.array(diff_hand)
        diff_cam = np.array(diff_cam)
        print (video_name, np.mean(diff_hand)/10000, np.median(diff_hand)/10000, np.min(diff_hand)/10000, np.max(diff_hand)/10000) # in milliseconds

        if args.timediff is not None:
            with open(os.path.join(args.timediff, f'{video_name}_timediff.json'), 'w') as f:
                json.dump(time_diff, f)
            continue

        # extract coarse and fine actions from annotated clips
        curr_idx = name2idx[video_name]
        curr_item = annot[curr_idx]['events']
        fps = annot[curr_idx]['videoMetadata']['video']['fps']
        actions = []
        action_list = []
        for i in range(len(curr_item)):
            if 'Coarse' in curr_item[i]['label']:
                coarse = curr_item[i]['attributes']['Action sentence']
            if 'Fine' in curr_item[i]['label'] and 'Correct' in curr_item[i]['attributes']['Action Correctness']:
                start_time = max(0, int(curr_item[i]['start']*fps))
                end_time = min(int(curr_item[i]['end']*fps), frame_num)
                actions.append((curr_item[i]['attributes']['Verb'], curr_item[i]['attributes']['Noun'], start_time, end_time, coarse))
                if 'adverbial' in curr_item[i]['attributes']:
                    hand_type = curr_item[i]['attributes']['adverbial']
                else:
                    hand_type = 'none'
                action_dict = {'verb': curr_item[i]['attributes']['Verb'], 'noun': curr_item[i]['attributes']['Noun'], 
                                'start': start_time, 'end': end_time, 'coarse': coarse, 'hand': hand_type}
                action_list.append(action_dict)
        
        pose_dict = {}

        for frame in tqdm.tqdm(range(num_frames)):
            if args.frame_num != None and frame != args.frame_num:
                continue
            
            hand_point_left = left_hand_array[img2hand_timestamps[frame]][4:].reshape(3, -1)
            hand_point_right = right_hand_array[img2hand_timestamps[frame]][4:].reshape(3, -1)

            img_pose = img_pose_array[img2cam_timestamps[frame]][2:].reshape(4, 4)

            ''' calculate extrinsics first and apply coordinate system transformation
            '''

            curr_hand_pose = {'left_world': hand_point_left.transpose(1,0).copy(), 
                              'right_world': hand_point_right.transpose(1,0).copy(), 
                              'axis_transform': axis_transform.copy()}
            
            # hand pose to the camera coordinate.
            hand_point_right = np.dot(axis_transform, np.dot(np.linalg.inv(
                img_pose), np.concatenate((hand_point_right, [[1]*np.shape(hand_point_right)[1]]))))
            hand_point_left = np.dot(axis_transform, np.dot(np.linalg.inv(
                img_pose), np.concatenate((hand_point_left, [[1]*np.shape(hand_point_right)[1]]))))
            
            # Put an empty camera pose for image.
            rvec = np.array([[0.0, 0.0, 0.0]])
            tvec = np.array([0.0, 0.0, 0.0])

            curr_hand_pose['left_cam'] = hand_point_left.transpose(1,0).copy()
            curr_hand_pose['right_cam'] = hand_point_right.transpose(1,0).copy()
            curr_hand_pose['cam2world'] = img_pose.copy()
            pose_dict[frame] = curr_hand_pose

            # For eyes
            if args.eye:
                point = get_eye_gaze_point(gaze_array[img2eye_timestamps[frame]], args.eye_dist) # og: gaze_indices[0][0]
                
                point_transformed = np.dot(axis_transform, np.dot(np.linalg.inv(
                    img_pose), np.concatenate((point, [1]))))

                img_points_gaze, _ = cv2.projectPoints(
                    point_transformed[:3].reshape((1, 3)), rvec, tvec, img_intrinsics, np.array([]))
                eyeproj_list.append(img_points_gaze[0][0])
                
            if args.save_video is not None:
                # Blue color in BGR
                img = cv2.imread(os.path.join(mpeg_img_path, '{0:06d}.png'.format(frame))) # og: mpeg_aligned_img_path  
                img_points_left, _ = cv2.projectPoints(
                    hand_point_left[:3], rvec, tvec, img_intrinsics, np.array([]))
                img_points_right, _ = cv2.projectPoints(
                    hand_point_right[:3], rvec, tvec, img_intrinsics, np.array([]))
                connectivity = get_handpose_connectivity()
                color = (255, 0, 0)
                radius = 5
                thickness = 2

                points = img_points_left
                #print("points",points)
                if not (np.isnan(points).any()):
                    for limb in connectivity:
                        cv2.line(img, (int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
                                (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])), color, thickness)
                color = (0, 255, 0)

                points = img_points_right
                if not (np.isnan(points).any()):
                    for limb in connectivity:
                        cv2.line(img, (int(points[limb[0]][0][0]), int(points[limb[0]][0][1])),
                                (int(points[limb[1]][0][0]), int(points[limb[1]][0][1])), color, thickness)

                if args.eye:
                    
                    points = img_points_gaze
                    if not np.isnan(points[0][0]).any():
                        color = (0, 0, 255)
                        thickness = 4
                        radius = 10
                        cv2.circle(img, (int(points[0][0][0]), int(
                            points[0][0][1])), radius, color, thickness)
                    #filename = os.path.join(projected_path, '{0:06d}.png'.format(frame))

                video_out.write(img)
        
        video_dict = {'actions': action_list, 'pose': pose_dict}
        
        if args.save_video is not None:
            cv2.destroyAllWindows()
            video_out.release()

        # save video action data in pickle format
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if args.save_action is not None:
            if not os.path.exists(args.save_action):
                os.makedirs(args.save_action)
            with open(os.path.join(args.save_action, f'{video_name}_action.pkl'), 'wb') as f:
                pickle.dump(video_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print (f"Saved {video_name} action data at {args.save_action}")
        
        # delete video dict
        del video_dict

        if args.eye and args.save_eyeproj:
            with open(os.path.join(base_path, "Eyes",'Eyes_proj.txt'), 'w') as f:
                for ii, elems in enumerate(eyeproj_list):
                    f.write(f"{gaze_timestamp[ii,0]}\t {gaze_timestamp[ii,1]}\t")
                    for elem in elems:
                        f.write(f"{elem}\t")
                    f.write("\n")

if __name__ == '__main__':
    main()
