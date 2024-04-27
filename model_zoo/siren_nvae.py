import torch
import torch.nn as nn
from model_zoo.siren import Siren

class MyArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value) 
class MSiren(torch.nn.Module):
    def __init__(self, routing_mode='0') -> None:
        super().__init__()
        args = MyArgs()
        args.pos_emb = 'ffm'
        args.ffm_map_scale = 16
        args.ffm_map_size = 64
        args.num_layers= 3
        args.hidden_dim = 256
        args.act_type = 'sine'
        args.dropout = False
        args.batch_size = 1
        self.routing_mode = routing_mode
        self.decoder = Siren(args=args, in_features=2, out_features=3, outermost_linear=True)

        if self.routing_mode == '1':
            self.latent_embedding1 = torch.nn.Linear(512//2, 256)
            self.latent_upsample1 = torch.nn.Upsample(scale_factor=8, mode='nearest')
            self.latent_embedding2 = torch.nn.Linear(256//2, 256)
            self.latent_upsample2 = torch.nn.Upsample(scale_factor=4, mode='nearest')
            self.latent_embedding3 = torch.nn.Linear(128//2, 256)
            self.latent_upsample3 = torch.nn.Upsample(scale_factor=2, mode='nearest')
            self.adaptive_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))

            self.router1 = nn.Sequential(
                nn.Conv2d(512//2,1,1,1,0),
                nn.ReLU(),
                nn.Conv2d(1,1,3,1,1),
                nn.AdaptiveAvgPool2d((1,1)),
                # nn.Sigmoid()
            )
            self.router2 = nn.Sequential(
                nn.Conv2d(256//2,1,1,1,0),
                nn.ReLU(),
                nn.Conv2d(1,1,3,1,1),
                nn.AdaptiveAvgPool2d((1,1)),
                # nn.Sigmoid()
            )
            self.router3 = nn.Sequential(
                nn.Conv2d(128//2,1,1,1,0),
                nn.ReLU(),
                nn.Conv2d(1,1,3,1,1),
                nn.AdaptiveAvgPool2d((1,1)),
                # nn.Sigmoid()
            )

    def forward(self, coords, dist_shape, latents=None, return_latent = False, return_routings=False):
        assert self.routing_mode in ['0', '1']
        b,c,h,w = dist_shape

        pos_embedding = self.decoder.map(coords)
        ac0 = self.decoder.net.net[0](pos_embedding)
        o1 = self.decoder.net.net[1][0](ac0)
        if latents is not None:
            latent1 = self.latent_embedding1(latents[5][0].permute(0,2,3,1)).permute(0,3,1,2)
            latent1 = self.latent_upsample1(latent1)
            latent1 = latent1.squeeze().unsqueeze(1)
            latent1 = latent1.view(b, 256, -1).permute(0, 2, 1)
            o1 = o1.unsqueeze(0) + latent1
        ac1 = self.decoder.net.net[1][1](o1)

        # routing method 1
        if self.routing_mode == '1':
            routing1 = self.router1(latents[5][0]).squeeze()
            routing1 = torch.sigmoid(routing1)
            ac1 = ac1 * routing1.unsqueeze(-1).unsqueeze(-1) + (o1-latent1) * (1 - routing1).unsqueeze(-1).unsqueeze(-1)

        o2 = self.decoder.net.net[2][0](ac1)
        if latents is not None:
            latent2 = self.latent_embedding2(latents[15][0].permute(0,2,3,1)).permute(0,3,1,2)
            latent2 = self.latent_upsample2(latent2)
            latent2 = latent2.view(b, 256, -1).permute(0, 2, 1)
            o2 = o2 + latent2
        ac2 = self.decoder.net.net[2][1](o2)

        # routing method 1
        if self.routing_mode == '1':
            routing2 = self.router2(latents[15][0]).squeeze()
            routing2 = torch.sigmoid(routing2)
            ac2 = ac2 * routing2.unsqueeze(-1).unsqueeze(-1) + (ac1) * (1 - routing2).unsqueeze(-1).unsqueeze(-1)

        o3 = self.decoder.net.net[3][0](ac2) 
        if latents is not None:
            latent3 = self.latent_embedding3(latents[35][0].permute(0,2,3,1)).permute(0,3,1,2)
            latent3 = self.latent_upsample3(latent3)
            latent3 = latent3.view(b, 256, -1).permute(0, 2, 1)
            o3 = o3 + latent3
        ac3 = self.decoder.net.net[3][1](o3)
        # routing method 1
        if self.routing_mode == '1':            
            routing3 = self.router3(latents[35][0]).squeeze()
            routing3 = torch.sigmoid(routing3)
            ac3 = ac3 * routing3.unsqueeze(-1).unsqueeze(-1) + (ac2) * (1 - routing3).unsqueeze(-1).unsqueeze(-1)
        ac4 = self.decoder.net.net[4](ac3)

        # routing method 1
        if self.routing_mode == '1':
            selected_routes = torch.stack([routing1, routing2, routing3], dim=0)

        ac4 = ac4.view(b, h, w, 3).permute(0, 3, 1, 2)
        if return_routings:
            if self.routing_mode == '1':
                return ac4, selected_routes
            if self.routing_mode == '0':
                return ac4, None
        return ac4, None