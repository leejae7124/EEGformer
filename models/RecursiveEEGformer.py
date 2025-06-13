# This is the script of EEG-Deformer
# This is the network script
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        dense_ip = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x = ff(x_cg) + x_fg
            dense_feature.append(x)

            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_ip.append(x_info)
            # x_dense = torch.cat(dense_ip, dim=-1)
        return dense_feature, dense_ip

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class Fusion(nn.Module):
    def __init__(self, input_channels):
        super(Fusion, self).__init__()
        self.weight = nn.Conv1d(input_channels, 1, kernel_size=1, bias=True)

    def forward(self, x, y):
        z = x.transpose(1, 2)  # (B, D, N)
        add_weight = torch.sigmoid(self.weight(z))  # (B, 1, N)
        add_weight = add_weight.transpose(1, 2)      # (B, N, 1)
        out = add_weight * x + (1 - add_weight) * y
        return out

class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan)

        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        # add
        self.linear1 = nn.Linear(500, 250)
        self.linear3 = nn.Linear(125, 250)
        self.linear4 = nn.Linear(62, 250)

        self.fusion1 = Fusion(250)
        self.fusion2 = Fusion(250)
        self.fusion3 = Fusion(250)
        self.fusion4 = Fusion(250)
        # end

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)
    
        self.mlp_head = nn.Sequential(
            nn.Linear(64256, num_classes)
        )

    def forward(self, eeg):
        # eeg: (b, chan, time)
        eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        x_list, ip_list = self.transformer(x)

        #RFP1
        r1, r2, r3, r4 = x_list #(500, 250, 125, 62)
        ip1, ip2, ip3, ip4 = ip_list
        # ip1 = self.IP_projection(ip1)    # (B, 256)  예: Linear(64→256)
        # ip2 = self.IP_projection(ip2)
        # ip3 = self.IP_projection(ip3)
        # ip4 = self.IP_projection(ip4)

        r1 = self.linear1(r1)
        r3 = self.linear3(r3)
        r4 = self.linear4(r4)

        t4 = r4
        t3 = r4 + r3
        t2 = t3 + r2 + r4
        t1 = t2 + r1 + r4

        #RFP2
        f1 = r1 + t1
        f2 = r2 + t2
        f3 = r3 + t3
        f4 = r4 + t4

        d4 = f4
        d3 = f4 + f3
        d2 = d3 + f2 + f4
        d1 = d2 + f1 + f4

        s1 = self.fusion1(d1, t1)
        s2 = self.fusion2(d2, t2)
        s3 = self.fusion3(d3, t3)
        s4 = self.fusion4(d4, t4)

        #RFP3
        p1 = r1 + s1
        p2 = r2 + s2
        p3 = r3 + s3
        p4 = r4 + s4

        d4 = p4
        d3 = p4 + p3
        d2 = d3 + p2 + p4
        d1 = d2 + p1 + p4

        o1 = self.fusion1(d1, s1)
        o2 = self.fusion2(d2, s2)
        o3 = self.fusion3(d3, s3)
        o4 = self.fusion4(d4, s4)

        # 여기서 각각 대응되는 ip 와 concat
        

        

        o1 = o1.view(o1.size(0), -1)
        o2 = o2.view(o2.size(0), -1)
        o3 = o3.view(o3.size(0), -1)
        o4 = o4.view(o4.size(0), -1)

        o1 = torch.cat([o1, ip1], dim=-1)   # (B, 256+256)
        o2 = torch.cat([o2, ip2], dim=-1)
        o3 = torch.cat([o3, ip3], dim=-1)
        o4 = torch.cat([o4, ip4], dim=-1)
        z = torch.cat([o1, o2, o3, o4], dim=-1)
        return self.mlp_head(z)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    data = torch.ones((16, 32, 1000))
    emt = Deformer(num_chan=32, num_time=1000, temporal_kernel=11, num_kernel=64,
                 num_classes=2, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.5)
    print(emt)
    print(count_parameters(emt))

    out = emt(data)
