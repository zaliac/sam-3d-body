
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContactHead(nn.Module):

    def __init__(
            self,
            in_channels=1280,
            hidden_dim=256,
            num_gcn_layers=3
    ):
        super().__init__()

        # reduce vertex feature
        # self.reduce = nn.Linear(in_channels, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d((1))
        # cross attention
        self.cross_attn = Cross_Att(480, 480)       # TODO

        # classifier
        self.classifier = Classifier(480)

    def forward(self, feat_map, verts_uv, adjacency=None):
        """
        feat_map : (B,1280,32,32)
        verts_uv : (B,6890,2)
        adjacency: (6890,6890)
        """

        B = feat_map.shape[0]

        # ------------------------------------------------
        # 1. vertex feature sampling
        # ------------------------------------------------
        # v_feat = self.sample_vertex_features(
        #     feat_map,
        #     verts_uv
        # )  # (B,6890,1280)

        # ------------------------------------------------
        # 2. feature reduction
        # ------------------------------------------------
        # v_feat = self.reduce(v_feat)  # (B,6890,256)

        # ------------------------------------------------
        # 3. image tokens
        # ------------------------------------------------
        # img_tokens = feat_map.flatten(2).permute(0, 2, 1)   # (B,1024,1280)
        # (B,1024,1280)




        # ------------------------------------------------
        # 4. cross attention
        # ------------------------------------------------
        # v_feat = self.cross_attn(
        #     v_feat,
        #     img_tokens
        # )  # (B,6890,256)

        # TODO: change the feat_map verts_uv

        att = self.cross_attn(feat_map, verts_uv)

        # ------------------------------------------------
        # 6. classifier
        # ------------------------------------------------
        # logits = self.classifier(v_feat).squeeze(-1)
        logits = self.classifier(att)





        # prob = torch.sigmoid(logits)
        #
        # return prob

        # Remove sigmoid, return logits for BCEWithLogitsLoss
        return logits


class Self_Attn(nn.Module):
    """ Self attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, out_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height = q.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(q.permute(0, 2, 1))
        # proj_query: reshape to B x c x N, N = H x W
        proj_key = self.key_conv(k.permute(0, 2, 1))
        # transpose check, energy: B x N x N, N = H x W
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(v.permute(0, 2, 1))
        # out: B x C x N
        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, C, height)
        out = out / np.sqrt(self.channel_in)

        return out


class Cross_Att(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Cross_Att, self).__init__()

        self.cross_attn_1 = Self_Attn(in_dim, out_dim)
        self.cross_attn_2 = Self_Attn(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm([1, in_dim])

    def forward(self, sem_seg, part_seg):
        cross1 = self.cross_attn_1(sem_seg, part_seg, part_seg)
        cross2 = self.cross_attn_1(part_seg, sem_seg, sem_seg)

        out = cross1 * cross2
        out = self.layer_norm(out)

        return out



class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim=6890):
        super(Classifier, self).__init__()

        self.out_dim = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 4096, True),
            nn.ReLU(),
            nn.Linear(4096, out_dim, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.classifier(x)
        return out.reshape(-1, self.out_dim)

# test
def test():
    B = 2

    feat_map = torch.randn(B,1280,32,32)
    verts_uv = torch.randn(B,6890,2)

    adjacency = torch.randn(6890,6890)

    model = ContactHead()

    prob = model(
        feat_map,
        verts_uv,
        adjacency
    )

    print(prob.shape)