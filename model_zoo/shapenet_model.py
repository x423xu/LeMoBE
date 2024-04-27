
import torch
import torch.nn as nn
from NVAE.model import AutoEncoder
from SnowflakeNet.models.model_ae import PointNetEncoder, SeedGenerator
from NVAE.distributions import Normal
# from NVAE.neural_operations import Conv2D

'''
modify preprocess to match shapnet input with nvae input
modify postprocess to match shapnet output with nvae output.
Here the ShapenetAutoEncoder is a small Autoencoder.
'''

class Stem(nn.Module):
    def __init__(self, args):
        super(Stem, self).__init__()
        # step 1: we want to transform the point cloud BxNx3 to BxNxd, then to Bxdxhxw
        self.encoder = PointNetEncoder(zdim=args.pc_encoder_dim)
        self.shape_embedding_m = SeedGenerator(dim_feat=args.pc_encoder_dim, num_pc=args.num_points)
        self.shape_embedding_v = SeedGenerator(dim_feat=args.pc_encoder_dim, num_pc=args.num_points)
        self.reshape = [32,64]
        self.upsample = nn.Upsample(size=(64,64), mode='nearest')

    def forward(self, x):
        c, v = self.encoder(x) # c is shape code Bx128, v is embedding: Bx128
        c_l = self.shape_embedding_m(c.unsqueeze(-1)).permute(0, 2, 1).contiguous()  # (B, 2048, 128)
        v_l = self.shape_embedding_v(v.unsqueeze(-1)).permute(0, 2, 1).contiguous()  # (B, 2048, 128)
        pc_feature = torch.cat([c_l, v_l], dim=2)  # (B, 2048, 256)
        B,N,D = pc_feature.shape

        assert N == self.reshape[0]*self.reshape[1] # make sure the number of points is correct
        pc_feature = pc_feature.permute(0, 2, 1).view(B,D,self.reshape[0],self.reshape[1]).contiguous()
        pc_feature = self.upsample(pc_feature) # Bx256x64x64
        return pc_feature

# class Stem(nn.Module):
#     def __init__(self, args) -> None:
#         super().__init__()
#         self.embed = nn.Linear(3,6)
#         # self.upsample = nn.Upsample(size=(64,64), mode='nearest')
#         self.transform = nn.Linear(args.num_points, 64*64)

#     def forward(self,x):
#         x = self.embed(x)
#         x = x.permute(0,2,1).contiguous()
#         x = self.transform(x)
#         b,n,c = x.shape
#         x = x.view(b,n,64,64)
#         return x

class ShapeNetAutoEncoder(AutoEncoder):
    def __init__(self, args, writer, arch_instance):
        self.args = args
        super(ShapeNetAutoEncoder, self).__init__(args, writer, arch_instance)
        self.stem = self.init_stem()
        self.post_processor = nn.Sequential(
                                            nn.Linear(64*64, args.num_points),
                                            nn.Tanh()
                                            )
        
        
    def init_stem(self):
        stem = Stem(self.args)
        return stem
    
    
    def forward(self, x):
        s = self.stem(x)

        # perform pre-processing
        for cell in self.pre_process:
            s = cell(s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = self.enc0(s)                            # this reduces the channel dimension
        param0 = self.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)   # for the first approx. posterior
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)

        # apply normalizing flows
        nf_offset = 0
        for n in range(self.num_flows):
            z, log_det = self.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += self.num_flows
        all_q = [dist]
        all_log_q = [log_q_conv]

        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0

        # prior for z0
        dist = Normal(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]

        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.res_dist else Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()
                    log_q_conv = dist.log_p(z)
                    # apply NF
                    for n in range(self.num_flows):
                        z, log_det = self.nf_cells[nf_offset + n](z, ftr)
                        log_q_conv -= log_det
                    nf_offset += self.num_flows
                    all_log_q.append(log_q_conv)
                    all_q.append(dist)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.vanilla_vae:
            s = self.stem_decoder(z)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)
        logits = self.post_processor(logits.view(logits.size(0), logits.size(1), -1))

        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            if self.with_nf:
                kl_per_var = log_q_conv - log_p_conv
            else:
                kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

        return logits, log_q, log_p, kl_all, kl_diag