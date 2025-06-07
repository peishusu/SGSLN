import torch
from torch import nn
from torch.cuda.amp import autocast
from enum import Enum
from models.CDXLSTM.vision_lstm import ViLBlock, SequenceTraversal
from torch.nn import functional as F
from functools import partial


from torchvision import models
import matplotlib.pyplot as plt
from timm.layers import DropPath, to_2tuple, trunc_normal_  # 新版本
from torchvision.models import VGG16_BN_Weights # torchvision 0.13+版本改变了加载预训练模型的方式
import math

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Atten_Cross(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Cross, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()

        # guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True) # map 2,1,128,128 -> 2,1,16,16
        #
        # guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) # query from x
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(cond)  # key from cond
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(cond) # value from cond
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

# self attention--spatial
class Atten_Spa(nn.Module):
    def __init__(self, in_dim):
        super(Atten_Spa, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [2,512,16,16] [2,1,128,128]
        m_batchsize, C, height, width = x.size()

        query = self.query_conv(x)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

class XLSTM_axial(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        # TODO ：尝试替换一下
        self.xlstm_h = ViLLayer(dim = in_channel)
        self.xlstm_w = ViLLayer(dim = in_channel)
        # self.xlstm_h = SKmLSTMLayer(dim=in_channel)
        # self.xlstm_w = SKmLSTMLayer(dim=in_channel)
        # SKmLSTMLayer
        self.xlstm_conv = conv_1x1(in_channel, in_channel)
        self.pos_emb_h = SqueezeAxialPositionalEmbedding(in_channel, 16)
        self.pos_emb_w = SqueezeAxialPositionalEmbedding(in_channel, 16)

    def forward(self, xA, xB):
        x_diff = xA - xB
        B,C,H,W = x_diff.shape
        pos_h = self.pos_emb_h(x_diff.mean(-1))
        pos_w = self.pos_emb_w(x_diff.mean(-2))
        x_xlstm_h = (self.xlstm_h(pos_h) + self.xlstm_h(pos_h.flip([-1])).flip([-1])).reshape(B, C, H, -1)
        x_xlstm_w = (self.xlstm_w(pos_w) + self.xlstm_w(pos_w.flip([-1])).flip([-1])).reshape(B, C, -1, W)
        x_xlstm = self.sigmoid(self.xlstm_conv(x_diff.add(x_xlstm_h.add(x_xlstm_w))))

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * x_xlstm * xA
        xB = B_weight * x_xlstm * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

class XLSTM_atten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

        # TODO：测试，我测试一下啊，！！！！ 代替换
        self.xlstm = ViLLayer(dim = in_channel)

        # self.xlstm = SKmLSTMLayer(dim = in_channel)


    def forward(self, xA, xB):
        x_diff = xA - xB
        B,C,H,W = x_diff.shape
        x_xlstm = (self.xlstm(x_diff) + self.xlstm(x_diff.flip([-1, -2])).flip([-1, -2]))

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * x_xlstm
        xB = B_weight * x_xlstm

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class ViLLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        out = x_vil.transpose(-1, -2).reshape(B, C, *img_dims)

        return out
def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)

        return x
