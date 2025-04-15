# How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions

## [Project Page](https://ap229997.github.io/projects/latentact) | [Paper](https://ap229997.github.io/projects/latentact/assets/paper.pdf) | [Supplementary](https://ap229997.github.io/projects/latentact/assets/suppmat.pdf) | [Video]() | [Poster]()

<p align="center">
  <img src="assets/teaser.png" height="256">
</p>

This repository contains the code for the CVPR 2025 paper [How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions](https://ap229997.github.io/projects/latentact/assets/paper.pdf). If you find our code or paper useful, please cite
```bibtex
@inproceedings{Prakash2025LatentAct,
                author = {Prakash, Aditya and Lundell, Benjamin and Andreychuk, Dmitry and Forsyth, David and Gupta, Saurabh and Sawhney, Harpreet},
                title = {How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions},
                booktitle = {Computer Vision and Pattern Recognition (CVPR)},
                year = {2025}
            }
```

## Setup

```
conda create --name latentact python=3.10
conda activate latentact

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

conda env update --file env.yml
```

The codebase also requires the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section. Since we only consider right hand, `MANO_RIGHT.pkl` is sufficient.

Set the environment variables:
```bash
export ROOT_DIR=<path_to_latentact_repo>
export MANO_PATH=<path_to_mano_models>
```

## Demo
Download pretrained model from `demo` folder in [link](https://drive.google.com/drive/folders/1u807hfuNgN7ZvJp5C_tk9GEzwoBlk40t?usp=sharing). It contains:
```
├── demo
│   ├── examples
│       ├── images...
│   ├── model
│       ├── indexer.tar
│       ├── interpred.tar
│       ├── opt.txt
│       ├── ranges.pkl

```

Check `demo.ipynb` for running demo. The notebook contains descriptions of each step.

## Data

We provide the annotations extracted from our [data engine](./data_engine). Check `downloads` folder in [link](https://drive.google.com/drive/folders/1u807hfuNgN7ZvJp5C_tk9GEzwoBlk40t?usp=sharing). It contains:
```
├── downloads
│   ├── holo_contact_dir
│   ├── holo_video_feats
│   ├── holo_video_inpaint_feats
│   ├── holo_settings_data_precomputed
│   ├── arctic_contact_dir
│   ├── arctic_video_feats
│   ├── arctic_data_precomputed
│   ├── holo_hand_bbox (optional)
│   ├── holo_hamer_preds (optional)
│   ├── holo_obj_masks (optional)
│   ├── holo_action_data (optional)
```

The last 4 annotations in the list are optional. They are generated from our data pipeline. The first 7 annotations correspond to different settings in our experiments. For the HoloAssist annotations, these are generated using the following scripts (these are not needed if you directly use the downloaded data).

**Preprocessing contact data**:
```bash
python preprocess_data.py --name debug --motion_type mano --code_dim 512 --nb_code 512 --batch_size 512 --contact_grid 16 --contact_dim 16 --pred_cam --contact_map --setting tasks
```
Change `--setting` from `tasks` (task-level) to `categories` (object-level), `actions` (action-level) or `instances` (scene-level) for the 4 generalization setting used in our work.

**Extracting video and text features**:
```bash
CUDA_VISIBLE_DEVICES=0 python compute_features.py --start_num 0 --end_num 10 --batch_size 4 --save_dir logs/debug_feats
```
The videos are processed in batches with `--start_num` and `--end_num` as indices of the video list.

## Training

Our framework involves a 2-stage training procedure: (1) Interaction Codebook: to learn a latent codebook of hand poses and contact points, i.e., tokenizing interaction trajectories, (2) a learned Indexer & an Interaction Predictor module to predict the interaction trajectories from single image, action text & 3D contact point. We use pretrained features for images (from DeiT) and text (from CLIP). The contact point is processed using the script above.

**Learning the codebook using VQVAE**:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --name debug_codebook --code_dim 512 --nb_code 512 --motion_type mano --contact_grid 16 --contact_dim 16 --video_feats 768 --text_feats --pred_cam --contact_map --batch_size 512 --setting tasks
```
Change `--setting` from `tasks` (task-level) to `categories` (object-level), `actions` (action-level) or `instances` (scene-level) for the 4 generalization setting used in our work.

Pass each motion sequence through the encoder and get the nearest codebook index:
```bash
CUDA_VISIBLE_DEVICES=0 python compute_indices.py --name debug_codebook --code_dim 512 --nb_code 512 --motion_type mano --contact_grid 16 --contact_dim 16 --video_feats 768 --text_feats --pred_cam --contact_map --batch_size 512 --return_indices --setting tasks
```

**Training the indexer**:

Indexer predicts a probability distribution over the codebook indices:
```bash
CUDA_VISIBLE_DEVICES=0 python train_prior.py --name debug_indexer --contact_grid 16 --pred_cam --contact_map --video_feats 768 --text_feats --batch_size 512 --decoder_only --setting tasks --only_first --load_indices debug_codebook
```
`--only_first` correspond to the forecasting setting. Change `--only_first` to `--interpolate` for interpolation setting and add `--use_inpaint` for the setting with hands absent in the image.

**Training the interaction predictor**:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --name debug_interpred --code_dim 512 --nb_code 512 --motion_type mano --contact_grid 16 --contact_dim 16 --video_feats 768 --text_feats --pred_cam --contact_map --batch_size 512 --setting tasks --decoder_only --only_first --load_indices debug_codebook
```

**Running on ARCTIC**:

Add `--dataset_name arctic` to run on [ARCTIC](https://arctic.is.tue.mpg.de/) dataset. We do not have 4 generalization settings on ARCTIC since the dataset is limited in scale of object interactions (omit `--setting` when running on ARCTIC). Note that we only show results for forecasting setting with hand in the paper. The above procedure can replicated for different settings on ARCTIC as well. The text descriptions and action segments are taken from [Text2HOI](https://github.com/JunukCha/Text2HOI).

## Evaluation

This requires the learned codebook, indexer and interpred checkpoints (check `demo` folder for a sample checkpoint). The learned codebook is already stored in the interpred checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --name debug_interpred --motion_type mano --code_dim 512 --nb_code 512 --contact_grid 16 --contact_dim 16 --pred_cam --contact_map --video_feats 768 --text_feats --batch_size 512 --decoder_only --only_first --load_prior debug_indexer --setting tasks
```

Change `--setting` from `tasks` (task-level) to `categories` (object-level), `actions` (action-level) or `instances` (scene-level) for the 4 generalization setting used in our work. Change `--only_first` to `--interpolate` for interpolation setting and add `--use_inpaint` for the setting with hands absent in the image.

## License

All the material here is released under the Creative Commons Attribution-NonCommerial 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc/4.0/). This means that you must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. You may not use the material for commercial purposes. 

For all the datasets and codebase (below) used in this work, refer to the respective websites/repos for citation and license details.

## Acknowledgements

This repo is built on top of the codebase from [Generative Masked Modeling of 3D Human Motions](https://github.com/EricGuo5513/momask-codes). We also modify code from several repositories for both the train/eval framework and data pipeline. We thank the authors for their work and for making their code available. Please check their respective repos for citation, licensing and usage.
- [MoMask](https://github.com/EricGuo5513/momask-codes)
- [MDM](https://github.com/GuyTevet/motion-diffusion-model)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [Text2HOI](https://github.com/JunukCha/Text2HOI)
- [PyHoloAssist](https://github.com/taeinkwon/PyHoloAssist)
- [Hands23](https://github.com/EvaCheng-cty/hands23_detector)
- [HaMeR](https://github.com/geopavlakos/hamer)
- [SAMv2](https://github.com/facebookresearch/sam2)
- [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS)
- [Affordance Diffusion](https://github.com/NVlabs/affordance_diffusion)
- [ARCTIC](https://github.com/zc-alexfan/arctic)