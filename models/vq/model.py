import os
import torch
import torch.nn as nn
from models.vq.encdec import *
from models.vq.residual_vq import ResidualVQ


class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 **kwargs):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        
        self.opt = args
        
        if self.opt is None or not hasattr(self.opt, 'model_type'):
            model_type = 'conv'
        else:
            model_type = self.opt.model_type
        if 'conv' in model_type:
            self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
            self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        elif 'tf' in model_type:
            self.encoder = TFEncoder(self.opt)
            self.decoder = TFDecoder(self.opt)
        
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

        if 'mano' in self.opt.motion_type:
            from smplx import MANO
            MANO_PATH = os.environ['MANO_PATH']
            self.mano = MANO(MANO_PATH, use_pca=False, flat_hand_mean=True)
            self.mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        if self.opt.pred_cam:
            # handle cases with unknown camera transformation (data is noisy)
            self.unknown_cam_transf = nn.Parameter(torch.randn(6+3))

        if self.opt.feedforward:
            assert not self.opt.decoder_only
            self.feedforward = FeedForward(self.opt)
            self.freeze_modules(encoder=True, decoder=True, quantizer=True, codebook=True)

        if self.opt.eval_model is not None:
            eval_dataset, eval_model = self.opt.eval_model.split('/')
            ckpt_path = os.path.join(self.opt.checkpoints_dir, eval_dataset, eval_model)
            self.load_vqvae_model(ckpt_path, encoder=False, decoder=True, quantizer=True)
            self.freeze_modules(encoder=True, decoder=True, quantizer=True, codebook=True)
        
        if self.opt.decoder_only:
            if self.opt.load_indices is None:
                print (f'No codebook provided, initializing with random codebook')
            else:
                if self.opt.transfer_from is None:
                    ckpt_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.load_indices)
                else:
                    splits = self.opt.transfer_from.split('/')
                    transfer_dataset, transfer_model = splits[0], splits[1]
                    if transfer_dataset == 'prior': # hardcoded hack for prior, only when transferring models across datasets, TODO: fix this
                        transfer_dataset = 'holo'
                        if 'final' in transfer_model: # hardcoder hack
                            if 'base' in transfer_model:
                                transfer_model = transfer_model.replace('_cmap', '')
                        else:
                            transfer_model = f'{transfer_model}_cmap'
                    ckpt_path = os.path.join(self.opt.checkpoints_dir, transfer_dataset, transfer_model)
                    assert os.path.exists(ckpt_path), f'Checkpoint path {ckpt_path} does not exist'
                self.load_codebook(ckpt_path, mode='latest')

                self.freeze_modules(encoder=True, quantizer=True, codebook=True)

    def load_codebook(self, ckpt_path, mode='finest'):
        if os.path.isdir(ckpt_path):
            ckpt = torch.load(f'{ckpt_path}/model/{mode}.tar', map_location='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        # extract codebook parameters from checkpoint keys
        codebook_keys = [k for k in ckpt['vq_model'].keys() if 'codebook' in k]
        codebook_ckpt = {k.replace('quantizer.', ''): ckpt['vq_model'][k] for k in codebook_keys}
        self.quantizer.load_state_dict(codebook_ckpt)
        print (f'Loaded codebook from {ckpt_path}')

    def load_vqvae_model(self, ckpt_path, encoder=False, decoder=False, quantizer=False, mode='finest', feedforward=False):
        if os.path.isdir(ckpt_path):
            ckpt = torch.load(f'{ckpt_path}/model/{mode}.tar', map_location='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        # load encoder, decoder, quantizer
        if encoder:
            encoder_keys = [k for k in ckpt['vq_model'].keys() if 'encoder' in k]
            encoder_ckpt = {k.replace('encoder.', ''): ckpt['vq_model'][k] for k in encoder_keys}
            self.encoder.load_state_dict(encoder_ckpt)
        if decoder:
            decoder_keys = [k for k in ckpt['vq_model'].keys() if 'decoder' in k]
            decoder_ckpt = {k.replace('decoder.', ''): ckpt['vq_model'][k] for k in decoder_keys}
            self.decoder.load_state_dict(decoder_ckpt)
        if feedforward:
            ff_keys = [k for k in ckpt['vq_model'].keys() if 'feedforward' in k]
            ff_ckpt = {k.replace('feedforward.', ''): ckpt['vq_model'][k] for k in ff_keys}
            self.feedforward.load_state_dict(ff_ckpt)
        if quantizer:
            self.load_codebook(ckpt_path)
    
    def freeze_modules(self, encoder=False, decoder=False, quantizer=False, codebook=False):
        if encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        if quantizer:
            for param in self.quantizer.parameters():
                param.requires_grad = False
        if codebook:
            for i in range(self.opt.num_quantizers):
                self.quantizer.layers[i].codebook.requires_grad = False
    
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        return code_idx, all_codes

    def forward(self, x, **kwargs):
        bz, ts, ch = x.shape
        more_outs = {}
        if not self.opt.decoder_only and not self.opt.feedforward:
            if ch == self.opt.dim_pose:
                x = x.reshape(bz, -1, self.opt.dim_pose)
            else:
                x = x.reshape(bz, ts, -1)
            x_in = self.preprocess(x)
            x_encoder = self.encoder(x_in)
            x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=self.opt.sample_codebook_temp)
        else:
            # hardcoded dummy values so that logging doesn't break
            commit_loss = torch.tensor(0).to(x.device)
            perplexity = torch.tensor(0).to(x.device)
            
            if self.opt.feedforward: # dummy values
                x_quantized = None
                code_idx = None
            else:
                # load code_idx from kwargs
                code_idx = kwargs.get('code_idx', None)
                assert code_idx is not None
                # get embeddings from quantizer
                x_quantized = self.quantizer.get_codebook_entry(code_idx.reshape(bz, ts, -1))
        
        if self.opt.feedforward:
            x_out = self.feedforward(x_quantized, **kwargs)
            x_out = x_out.reshape(bz, ts, -1)
        
        else:
            ## decoder
            x_out = self.decoder(x_quantized, **kwargs)
            x_out = x_out.reshape(bz, ts, -1) # (bs, T, Jx3)
            # x_out = self.postprocess(x_decoder)
        
        if self.opt.return_indices:
            more_outs['code_idx'] = code_idx

        return x_out, commit_loss, perplexity, more_outs

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)