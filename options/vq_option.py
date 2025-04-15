import argparse
import os
from packaging.version import parse
import torch

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='holo', help='dataset directory')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=30, help='training motion length')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')

    ## optimization
    parser.add_argument('--max_epoch', default=50, type=int, help='number of total epochs to run')
    # parser.add_argument('--total_iter', default=None, type=int, help='number of total iterations to run')
    parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=3e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')

    parser.add_argument('--num_quantizers', type=int, default=6, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')
    # parser.add_argument('--use_vq_prob', type=float, default=0.8, help='quantize_dropout_prob')
    parser.add_argument('--sample_codebook_temp', type=float, default=0.5, help='gumbel softmax temperature')
    parser.add_argument('--ext', type=str, default='default', help='reconstruction loss')

    ## other
    parser.add_argument('--name', type=str, default="test", help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='./logs', help='models are saved here')
    parser.add_argument('--log_every', default=10, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=2, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=5, type=int, help='save eval results every n epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')

    ## For Res Predictor only
    parser.add_argument('--vq_name', type=str, default="rvq_nq6_dc512_nc512_noshare_qdp0.2", help='Name of this trial')
    # parser.add_argument('--n_res', type=int, default=2, help='Name of this trial')
    # parser.add_argument('--do_vq_res', action="store_true")
    parser.add_argument("--seed", default=0, type=int)

    # for LatentAct
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--motion_type", type=str, default="mano", help="motion type")
    parser.add_argument("--model_type", type=str, default="tf", help="model type")
    parser.add_argument("--n_freq", type=int, default=4, help="num of frequency components for positional encoding")
    parser.add_argument("--viz_every_e", type=int, default=1e12, help="viz every n epoch for validation")
    parser.add_argument("--viz_every_t", type=int, default=1e12, help="viz every n iters for training")
    parser.add_argument("--val_every_e", type=int, default=5, help="validation every n epoch")
    parser.add_argument("--num_workers", type=int, default=8, help="num of workers for dataloader")
    parser.add_argument("--num_threads", type=int, default=8, help="num of threads for torch")
    
    # different modalities
    parser.add_argument("--video_feats", type=int, default=None, help="video feature dimension")
    parser.add_argument("--text_feats", action="store_true", help="use text features")
    parser.add_argument("--contact_grid", type=int, default=None, help="use contact features")
    parser.add_argument("--contact_dim", type=int, default=16, help="contact feature dimension")
    parser.add_argument("--coord_sys", type=str, default=None, help="camera or hand or affordance")
    parser.add_argument("--pred_cam", action="store_true", help="predict camera rotation and translation")
    parser.add_argument("--joints_loss", action="store_true", help="use joints loss")
    parser.add_argument("--contact_map", action="store_true", help="use contact map")
    parser.add_argument("--residual_transf", action="store_true", help="use residual prediction")
    parser.add_argument("--return_indices", action="store_true", help="return codebook indices")
    
    # different settings
    parser.add_argument("--only_first", action="store_true", help="use first frame only as decoder input")
    parser.add_argument("--decoder_only", action="store_true", help="use first frame only as decoder input")
    parser.add_argument("--use_inpaint", action="store_true", help="use inpainted images with hands removed")
    parser.add_argument("--setting", type=str, default=None, help="which experiment setting to run")
    parser.add_argument("--interpolate", action="store_true", help="also provide goal image")
    parser.add_argument("--traj_only", action="store_true", help="autoencode trajectory only, no other inputs to decoder")
    parser.add_argument("--dataset_size", type=int, default=None, help="dataset size to use for training")
    
    # eval only params
    parser.add_argument("--inference", action="store_true", help="eval mode for inference")
    parser.add_argument("--stochastic", action="store_true", help="stochastic gumbel sampling")
    parser.add_argument("--save_name", type=str, default=None, help="save name for eval results file")
    parser.add_argument("--random", action="store_true", help="random sampling codebook indices")
    parser.add_argument("--infer_iter", type=int, default=1, help="number of samplings for inference")
    parser.add_argument("--eval_also", action="store_true", help="run eval also")
    parser.add_argument("--viz_gt", action="store_true", help="visualize ground truth data")
    parser.add_argument("--viz_pred", type=str, default=None, help="visualize prediction from given model")
    parser.add_argument("--load_prior", type=str, default=None, help="load prior model")
    parser.add_argument("--debug_viz", action="store_true", help="debug visualization")
    
    # preprocess params
    parser.add_argument("--save_contacts", action="store_true", help="preprocess and save contact info")
    parser.add_argument("--preprocess", action="store_true", help="preprocess and save features")
    parser.add_argument("--load_indices", type=str, default=None, help="where to load indices from")
    
    # feedforward params
    parser.add_argument("--feedforward", action="store_true", help="use single pass feedforward model")
    parser.add_argument("--use_vit", action="store_true", help="train ViT encoder as well")
    
    # cross-dataset transfer params
    parser.add_argument("--transfer_from", type=str, default=None, help="transfer from this pretrained model")
    parser.add_argument("--eval_model", type=str, default=None, help="evaluate this pretrained model")
    
    # evaluate text2hoi
    parser.add_argument("--text2hoi", action="store_true", help="evaluate text2hoi model")

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt