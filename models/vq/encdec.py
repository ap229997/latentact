import torch
import torch.nn as nn
from models.vq.resnet import Resnet1D
from models.vq.transformer import *


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None
                 ):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, **kwargs):
        video_feats = kwargs.get('video_feats', None)
        if video_feats is not None:
            x = x + video_feats
        x = self.model(x)
        return x.permute(0, 2, 1)
    

class PosEnc(nn.Module):
    def __init__(self, n_freq: int = 4) -> None:
        super().__init__()
        self.n_freq = n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor: # taken from nerfstudio
        # x: (B, 3, T) where 3 is (x, y, z) position in 3D space
        x_in = 2 * torch.pi * x # (B, 3, T)  2pi * x
        freqs = 2 ** torch.linspace(0, self.n_freq-1, self.n_freq).to(x.device)
        x_s = x_in.unsqueeze(1) * freqs.reshape(1, -1, 1, 1) # (B, n_freq, 3, T)
        x_scaled = x_s.reshape(x.shape[0], -1, x.shape[-1]) # (B, 3*n_freq, T)
        x_sin = torch.sin(x_scaled) # (B, 3*n_freq, T)
        x_cos = torch.cos(x_scaled) # (B, 3*n_freq, T)
        x_pos = torch.stack([x_sin, x_cos], dim=1) # (B, 2, 3*n_freq, T)
        x_pos = x_pos.reshape(x.shape[0], -1, x.shape[-1])# (B, 6*n_freq, T) sin and cos interleaved
        x_out = torch.cat([x, x_pos], dim=1) # (B, 3 + 6*n_freq, T)
        return x_out


class TFEncoder(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        if opt.dim_pose == 3:
            inp_pose = opt.dim_pose + 3*2*opt.n_freq
        else:
            inp_pose = opt.dim_pose

        if opt.contact_map:
            inp_pose = inp_pose + 778 # MANO vertices

        self.inp_head = nn.Sequential(nn.Linear(inp_pose, opt.code_dim), nn.ReLU()) # this is tf
        self.enc = TransformerEncoderLayer(d_model=opt.code_dim, nhead=1, batch_first=True, dropout=0)
        self.transformer = TransformerEncoder(self.enc, num_layers=1)
        self.pos_enc = PosEnc(n_freq=opt.n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T)
        if self.opt.dim_pose == 3:
            x = self.pos_enc(x) # (B, 3 + 6*n_freq, T)
        x = x.permute(0, 2, 1) # (B, T, 3 + 6*n_freq)
        x = self.inp_head(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return x
    

class TFDecoder(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        opt.tf_dim = opt.code_dim
        self.dec = TransformerDecoderLayer(d_model=opt.tf_dim, nhead=1, batch_first=True, dropout=0.2)
        self.transformer = TransformerDecoder(self.dec, num_layers=1)

        if 'joints' in opt.motion_type:
            out_dim = opt.joints_num*opt.dim_pose
        elif 'mano' in opt.motion_type:
            out_dim = opt.dim_pose
        else:
            raise ValueError('Unknown motion type')

        self.head = nn.Sequential(
                nn.Linear(opt.tf_dim, 128),
                nn.ReLU(), 
                nn.Linear(128, out_dim),
            )
        
        pos_enc = nn.Parameter(torch.randn(opt.window_size, opt.tf_dim))
        self.register_parameter('pos_enc', pos_enc)

        if not opt.traj_only and (opt.video_feats is not None or opt.text_feats or opt.contact_grid is not None):
            joint_dim = opt.code_dim
            if opt.video_feats is not None:
                joint_dim = joint_dim + opt.video_feats
            if opt.text_feats:
                joint_dim = joint_dim + 512 # clip features
            if opt.contact_grid is not None:
                self.contact_module = ContactModule(opt)
                joint_dim = joint_dim + self.contact_module.feat_dim * 2
            
            self.joint_mlp = nn.Sequential(
                    nn.Linear(joint_dim, opt.tf_dim),
                    nn.ReLU(),
                )
            
        if opt.contact_map:
            self.contact_map = ContactMap(opt)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bz, ch, tk = x.shape
        tgt = self.pos_enc.unsqueeze(0).expand(bz, -1, -1)
        memory = x.permute(0, 2, 1) # (B, T, 512)
        
        if not self.opt.traj_only and (self.opt.video_feats is not None or self.opt.text_feats or self.opt.contact_grid is not None):
            combined_feats = memory
            video_feats = kwargs.get('video_feats', None)
            if video_feats is not None:
                combined_feats = torch.cat([combined_feats, video_feats], dim=-1)
            text_feats = kwargs.get('text_feats', None)
            if text_feats is not None:
                combined_feats = torch.cat([combined_feats, text_feats], dim=-1)
            if self.opt.contact_grid is not None:
                contact_volume = kwargs.get('contact', None)
                assert contact_volume is not None
                grid_volume = kwargs.get('grid', None)
                assert grid_volume is not None
                contact_mask = kwargs.get('contact_mask', None)
                if contact_mask is None:
                    # assume all contacts are known
                    contact_mask = torch.ones(bz, tk).to(x.device)
                grid_mask = kwargs.get('grid_mask', None)
                if grid_mask is None:
                    grid_mask = torch.ones(bz).to(x.device)
                contact_feats = self.contact_module(contact_volume, grid_volume, contact_mask, grid_mask)
                if isinstance(contact_feats, tuple):
                    contact_feats, grid_feats = contact_feats
                    combined_feats = torch.cat([combined_feats, contact_feats, grid_feats], dim=-1)
                else:
                    combined_feats = torch.cat([combined_feats, contact_feats], dim=-1)
            
            memory = self.joint_mlp(combined_feats)
        
        output = self.transformer(tgt, memory) # (B, T, 512)
        motion_out = self.head(output) # (B, T, opt.dim_pose)
        if self.opt.contact_map:
            contact_map = self.contact_map(output) # (B, T, 778)
            motion_out = torch.cat([motion_out, contact_map], dim=-1) # (B, T, 778 + opt.dim_pose)
        return motion_out
    

class ContactModule(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.grid_size = opt.contact_grid
        self.contact_dim = opt.contact_dim

        # process NxNxN grid with 3D convolutions and downsample to 16x16x16
        self.grid_conv = nn.Sequential(
            nn.Conv3d(3, self.contact_dim//4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim//4, self.contact_dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim//2, self.contact_dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim//2, self.contact_dim, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim, self.contact_dim, 3, 1, 1),
            nn.ReLU(),
        )

        # keep both grid_conv and contact_conv of the same dimension
        self.contact_conv = nn.Sequential(
            nn.Conv3d(1, self.contact_dim//4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim//4, self.contact_dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim//2, self.contact_dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim//2, self.contact_dim, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(self.contact_dim, self.contact_dim, 3, 1, 1),
            nn.ReLU(),
        )

        downsample = 0
        for module in self.contact_conv:
            if isinstance(module, nn.MaxPool3d):
                downsample += 1
        
        self.feat_dim = self.contact_dim * ((self.grid_size // (2 ** downsample)) ** 3)
        # create a learnable embedding for unknown contacts
        self.unknown_contact = nn.Parameter(torch.randn((self.grid_size, self.grid_size, self.grid_size)))
        self.unknown_grid = nn.Parameter(torch.randn((3, self.grid_size, self.grid_size, self.grid_size)))
            

    def forward(self, c_vol: torch.Tensor, g_vol: torch.Tensor, c_mask: torch.Tensor, g_mask: torch.Tensor) -> torch.Tensor:
        # c_vol: (B, T, N, N, N)
        # c_mask: (B, T), 1 for known contacts, 0 for unknown contacts
        bz, ts, _, _, _ = c_vol.shape
        x = c_vol.view(bz*ts, self.grid_size, self.grid_size, self.grid_size)
        
        # replace unknown contacts with learnable embeddings
        c_known_mask = c_mask.view(bz*ts,1,1,1).expand(bz*ts, self.grid_size, self.grid_size, self.grid_size)
        x = x * c_known_mask + (1-c_known_mask) * self.unknown_contact.unsqueeze(0)

        g = g_vol.permute(0,4,1,2,3).contiguous() # (B, N, N, N, 3) -> (B, 3, N, N, N)
        g_known_mask = g_mask.view(bz,1,1,1,1).expand(bz, 3, self.grid_size, self.grid_size, self.grid_size)
        g = g * g_known_mask + (1-g_known_mask) * self.unknown_grid.unsqueeze(0)
        
        x = self.contact_conv(x.unsqueeze(1))
        x = x.view(bz, ts, -1) # (B, T, F*8*8*8)

        g = self.grid_conv(g) # (B*T, 1, 8, 8, 8)
        g = g.view(bz, 1, -1).expand(bz, ts, -1)
        return x, g
    

class ContactMap(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        # 778-way classifier
        self.fc = nn.Sequential(
            nn.Linear(opt.tf_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 778),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 512)
        out = self.fc(x) # logits, (B, T, 778)
        return out


class FeedForward(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        if self.opt.use_vit:
            # load vit model with pretrained weights
            from timm import create_model
            self.vit = create_model('deit_base_patch16_224', pretrained=True)
            self.vit.head = nn.Identity()

        opt.tf_dim = opt.code_dim
        self.dec = TransformerDecoderLayer(d_model=opt.tf_dim, nhead=4, batch_first=True, dropout=0.2)
        self.transformer = TransformerDecoder(self.dec, num_layers=4)
        
        if 'joints' in opt.motion_type:
            out_dim = opt.joints_num*opt.dim_pose
        elif 'mano' in opt.motion_type:
            out_dim = opt.dim_pose
        else:
            raise ValueError('Unknown motion type')

        self.head = nn.Sequential(
                nn.Linear(opt.tf_dim, 128),
                nn.ReLU(), 
                nn.Linear(128, out_dim),
            )
        
        pos_enc = nn.Parameter(torch.randn(opt.window_size, opt.tf_dim))
        self.register_parameter('pos_enc', pos_enc)

        assert (opt.video_feats is not None or opt.text_feats or opt.contact_grid is not None)
        joint_dim = 0
        if opt.video_feats is not None:
            joint_dim = joint_dim + opt.video_feats
        if opt.text_feats:
            joint_dim = joint_dim + 512 # clip features
        if opt.contact_grid is not None:
            self.contact_module = ContactModule(opt)
            joint_dim = joint_dim + self.contact_module.feat_dim * 2
        
        self.joint_mlp = nn.Sequential(
                nn.Linear(joint_dim, opt.tf_dim),
                nn.ReLU(),
                nn.Linear(opt.tf_dim, opt.tf_dim),
                nn.ReLU(),
                nn.Linear(opt.tf_dim, opt.tf_dim),
                nn.ReLU(),
            )
            
        if opt.contact_map:
            self.contact_map = ContactMap(opt)

    def forward(self, x=None, **kwargs) -> torch.Tensor:
        video_feats = kwargs.get('video_feats', None)
        assert video_feats is not None # video_feats is required
        bz = video_feats.shape[0]
        
        if self.opt.use_vit:
            ch, h, w = video_feats.shape[-3:]
            video_feats = video_feats.reshape(-1, ch, h, w)
            video_feats = self.vit(video_feats)
            video_feats = video_feats.reshape(bz, -1, self.opt.video_feats)
            video_feats = torch.sum(video_feats, dim=1, keepdim=True)
            video_feats = video_feats.repeat(1, self.opt.window_size, 1)
        
        tgt = self.pos_enc.unsqueeze(0).expand(bz, -1, -1)
        
        combined_feats = []
        combined_feats.append(video_feats)
        text_feats = kwargs.get('text_feats', None)
        if text_feats is not None:
            combined_feats.append(text_feats)
        if self.opt.contact_grid is not None:
            contact_volume = kwargs.get('contact', None)
            assert contact_volume is not None
            grid_volume = kwargs.get('grid', None)
            assert grid_volume is not None
            contact_mask = kwargs.get('contact_mask', None)
            if contact_mask is None:
                # assume all contacts are known
                contact_mask = torch.ones(bz, tk).to(x.device)
            grid_mask = kwargs.get('grid_mask', None)
            if grid_mask is None:
                grid_mask = torch.ones(bz).to(x.device)
            contact_feats = self.contact_module(contact_volume, grid_volume, contact_mask, grid_mask)
            if isinstance(contact_feats, tuple):
                contact_feats, grid_feats = contact_feats
                combined_feats.append(contact_feats)
                combined_feats.append(grid_feats)
            else:
                combined_feats.append(contact_feats)
        
        combined_feats = torch.cat(combined_feats, dim=-1)
        memory = self.joint_mlp(combined_feats) # (B, T, latent_dim)
        
        output = self.transformer(tgt, memory) # (B, T, 512)
        motion_out = self.head(output) # (B, T, opt.dim_pose)
        if self.opt.contact_map:
            contact_map = self.contact_map(output) # (B, T, 778)
            motion_out = torch.cat([motion_out, contact_map], dim=-1) # (B, T, 778 + opt.dim_pose)
        return motion_out