import torch
import torch.nn as nn

from models.vq.encdec import ContactModule


class LearnedPrior(nn.Module):
    def __init__(self, opt):
        super(LearnedPrior, self).__init__()
        self.opt = opt

        self.contact_module = ContactModule(opt)

        text_dim = 512 # CLIP embeddings, hardcoded for now
        if opt.use_contact_points:
            inp_dim = text_dim + opt.video_feats + self.contact_module.feat_dim * 2 + 3 # 3 for contact point
        else:
            inp_dim = text_dim + opt.video_feats + self.contact_module.feat_dim * 2
        out_dim = opt.nb_code
        latent_dim = 2 * opt.nb_code

        self.decoder_head = TFClassifier(opt, inp_dim)
        
        tf_out_dim = self.decoder_head.decoder.layers[-1].linear2.out_features
        self.classifier = nn.Sequential(
            nn.Linear(tf_out_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(opt.dropout),
            nn.Linear(latent_dim, out_dim)
        )

    def forward(self, text_feats, video_feats, contact_map, contact_mask, grid_map, grid_mask, contact_point):
        if len(contact_map.shape) == 4:
            c_vol = contact_map.unsqueeze(1) # add time dim to match ContactModule input
            c_mask = contact_mask.unsqueeze(1)
        else:
            c_vol = contact_map
            c_mask = contact_mask
        
        g_vol = grid_map
        g_mask = grid_mask

        contact_feats = self.contact_module(c_vol, g_vol, c_mask, g_mask)
        if isinstance(contact_feats, tuple):
            contact_feats, grid_feats = contact_feats
            contact_feats = torch.cat([contact_feats, grid_feats], dim=-1)
        contact_feats = contact_feats.squeeze(1) # remove time dim

        if self.opt.use_contact_points:
            feats = torch.cat([text_feats, video_feats, contact_feats, contact_point], dim=-1)
        else:
            feats = torch.cat([text_feats, video_feats, contact_feats], dim=-1)
        
        decoder_out = self.decoder_head(feats)
        logits = self.classifier(decoder_out)

        return logits
    

class TFClassifier(nn.Module):
    def __init__(self, opt, inp_dim):
        super(TFClassifier, self).__init__()
        self.opt = opt

        # classifier head using transfomer decoder
        ts, nq = opt.window_size, opt.num_quantizers

        self.pos_enc = nn.Parameter(torch.randn(1, ts*nq, opt.code_dim))

        self.joint_mlp = nn.Sequential(
            nn.Linear(inp_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, opt.code_dim),
            nn.ReLU()
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=opt.code_dim,
                nhead=1,
                dropout=opt.dropout,
                batch_first=True
            ),
            num_layers=1
        )

    def forward(self, memory):
        bz = memory.shape[0]
        tgt =  self.pos_enc.repeat(bz, 1, 1)
        memory = memory.unsqueeze(1)
        memory = self.joint_mlp(memory) 
        out = self.decoder(tgt, memory)

        return out