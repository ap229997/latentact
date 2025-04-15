import os
import random
from typing import Dict

def get_video_subset(annot, task_type):
    # these are manually curated to include relevant contact sequences
    rel_verbs = ['grab', 'place', 'lift', 'pull', 'close', 'open', 'screw', 'press', 'turn', 'mix/stir', 'rotate', 'align', 'hold', 'press', 'slide', 'push', 'flip', 'touch', 'load', 'lock']
    not_rel_nouns = ['hex_socket_head', 'screw', 'hexagonal_wrench', 'hexagonal_cap_nut', 'bolt', 'washer', 'hex_nut']
    relevant_frames = {}
    total_seqs = 0
    for i in range(len(annot)):
        for task in task_type:
            if task.lower() in annot[i]['video_name'].lower():
                video_name = annot[i]['video_name']
                assert video_name not in relevant_frames, f'Video {video_name} already in relevant_frames.'
                relevant_frames[video_name] = []
                fps = annot[i]['videoMetadata']['video']['fps']
                curr_events = annot[i]['events']
                for event in curr_events:
                    if 'Fine' in event['label'] and 'Correct' in event['attributes']['Action Correctness']:
                        noun = event['attributes']['Noun']
                        verb = event['attributes']['Verb']
                        # print (event['attributes'])
                        if 'adverbial' not in event['attributes']:
                            continue
                        hand_type = event['attributes']['adverbial']
                        start, end = int(event['start']*fps), int(event['end']*fps)
                        if verb in rel_verbs and noun not in not_rel_nouns:
                            if video_name not in relevant_frames:
                                relevant_frames[video_name] = []
                            relevant_frames[video_name].append((start, end, verb, noun, hand_type.split(' ')[0]))
                            total_seqs += 1

    # manually checked anomalies
    anomaly = ['z069-june-29-22-rashult_assemble', 'z093-july-11-22-rashult_assemble', 'z017-june-20-22-rashult_assemble', 
               'z019-june-20-22-rashult_assemble', 'z035-june-23-22-rashult_assemble', 'z038-june-23-22-rashult_assemble', 
               'z019-june-20-22-knarrevik_assemble', 'z047-june-25-22-rashult_assemble', 'z048-june-25-22-rashult_assemble', 
               'z051-june-27-22-rashult_assemble', 'z051-june-27-22-rashult_disassemble', 'z021-june-20-22-knarrevik_assemble',
               'z042-june-24-22-knarrevik_assemble', 'z053-june-27-22-rashult_assemble', 'z053-june-27-22-rashult_disassemble',
               'z055-june-27-22-knarrevik_assemble', 'z055-june-27-22-knarrevik_disassemble', 'z056-june-28-22-knarrevik_assemble',
               'z056-june-28-22-knarrevik_disassemble', 'z051-june-27-22-knarrevik_assemble', 'z051-june-27-22-knarrevik_disassemble',
               'z076-july-01-22-rashult_assemble', 'z076-july-01-22-rashult_disassemble']

    for ana in anomaly:
        if ana in relevant_frames:
            relevant_frames.pop(ana)
    task_str = ' '.join(task_type)
    print(f'Found {len(relevant_frames)} videos with {total_seqs} sequences across tasks {task_str}.')
    return relevant_frames


def sample_clips(video_dict: Dict[str, list], num_clips: int = None) -> list[Dict]:
    clip_data = []
    for k, v in video_dict.items():
        for clip in v:
            clip_data.append({'video_name': k, 'clip': clip})
    total_clips = sum([len(video_dict[k]) for k in video_dict])
    assert len(clip_data) == total_clips
    print (f'Found {total_clips} clips.')
    if num_clips is None:
        return clip_data
    sampled_clips = random.choices(clip_data, k=min(num_clips, total_clips))
    # compute total frames
    total_frames = sum([clip['clip'][1]-clip['clip'][0]+1 for clip in sampled_clips])
    print (f'Sampled {len(sampled_clips)} clips with total frames {total_frames}.')
    return sampled_clips