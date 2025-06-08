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

class ResidualBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ResidualBasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

        # 残差路径：是否需要 projection
        if in_planes != out_planes or stride != 1:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.residual_conv = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.residual_conv(x)
        out = self.conv(x)
        out = self.bn(out)
        out += identity  # 残差相加
        return self.relu(out)  # 残差连接后再激活



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

        # self.convA = nn.Conv2d(in_channel, 1, 1)
        # self.convB = nn.Conv2d(in_channel, 1, 1)
        # self.sigmoid = nn.Sigmoid()
        # ✅ 替换 convA/convB → CBAM
        self.cbamA = CBAMBlock(in_channel)
        self.cbamB = CBAMBlock(in_channel)

        self.xlstm_h = SKmLSTMLayer(dim=in_channel)
        self.xlstm_w = SKmLSTMLayer(dim=in_channel)

        self.xlstm_conv = conv_1x1(in_channel, in_channel)
        self.sigmoid = nn.Sigmoid()

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

        # A_weight = self.sigmoid(self.convA(x_diffA))
        # B_weight = self.sigmoid(self.convB(x_diffB))
        # ✅ 用 CBAM 替换 conv+sigmoid 权重
        A_weight = self.cbamA(x_diffA)
        B_weight = self.cbamB(x_diffB)

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

        # Conv模块
        # self.convA = nn.Conv2d(in_channel, 1, 1)
        # self.convB = nn.Conv2d(in_channel, 1, 1)
        # sigmod模块
        # self.sigmoid = nn.Sigmoid()
        # ✅ 替换掉 convA / convB + sigmoid
        self.cbamA = CBAMBlock(in_channel)
        self.cbamB = CBAMBlock(in_channel)

        # self.xlstm = ViLLayer(dim = in_channel)
        self.xlstm = SKmLSTMLayer(dim = in_channel)


    def forward(self, xA, xB):
        x_diff = xA - xB
        B,C,H,W = x_diff.shape

        # Bi-directional SKmLSTM 处理 diff 特征
        x_xlstm = (self.xlstm(x_diff) + self.xlstm(x_diff.flip([-1, -2])).flip([-1, -2]))

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        # A_weight = self.sigmoid(self.convA(x_diffA))
        # B_weight = self.sigmoid(self.convB(x_diffB))
        A_weight = self.cbamA(x_diffA)
        B_weight = self.cbamB(x_diffB)

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



'''
    所请求的 SK-Fusion + mLSTM 混合优化版本 的代码实现，结构和输入输出格式完全 对齐 ViLLayer（即：输入 [B, C, H, W] → 展平 → 处理 → 再 reshape 回 [B, C, H, W]）。
'''
class SKmLSTMLayer(nn.Module):
    def __init__(self, dim, hidden_dim=None, reduction=8):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim

        self.norm = nn.LayerNorm(dim)

        # 分支1：标准LSTM
        self.lstm = nn.LSTM(input_size=dim, hidden_size=self.hidden_dim, batch_first=True)

        # 分支2：简化 mLSTM
        self.W_m = nn.Linear(self.hidden_dim, dim, bias=False)
        self.m_lstm_gates = nn.Linear(dim, 4 * self.hidden_dim)

        # 分支3：跳连路径（轻量）
        self.skip_proj = nn.Identity()

        # 门控机制：Selective Kernel 融合权重学习
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim // reduction)
        self.fc2 = nn.Linear(self.hidden_dim // reduction, 3)
        self.softmax = nn.Softmax(dim=1)

        # 输出投影回 dim
        self.out_proj = nn.Linear(self.hidden_dim, dim)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()

        B, C = x.shape[:2]
        assert C == self.dim, f"Input channel {C} != self.dim {self.dim}"
        n_tokens = x.shape[2:].numel()
        spatial_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)  # [B, T, C]
        x_norm = self.norm(x_flat)

        # 分支1：标准LSTM
        h_lstm, _ = self.lstm(x_norm)  # [B, T, H]

        # 分支2：简化 mLSTM
        m = self.W_m(h_lstm) * x_norm  # 乘法门
        gates = self.m_lstm_gates(m)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_m = i * g  # 没有历史c，简化版
        h_m = o * torch.tanh(c_m)  # [B, T, H]

        # 分支3：跳连路径
        h_skip = self.skip_proj(x_norm)  # [B, T, C]

        # 融合准备
        h_concat = torch.stack([h_lstm, h_m, h_skip], dim=1)  # [B, 3, T, H]

        # 全局注意力权重
        # gap = h_concat.mean(dim=2).mean(dim=-1)  # [B, 3]
        gap = h_concat.mean(dim=2)  # ✅ 正确：只对时间维度做 GAP，结果是 [B, 3, HIDDEN_DIM]
        gap_flat = gap.reshape(B, -1)  # 得到 [B, 3 * HIDDEN_DIM]

        # gap_flat = torch.cat([gap[:, i] for i in range(3)], dim=-1)  # [B, 3H]
        attn = self.fc2(F.relu(self.fc1(gap_flat)))  # [B, 3]
        weights = self.softmax(attn).unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]

        # 加权融合
        fused = (h_concat * weights).sum(dim=1)  # [B, T, H]

        # 输出映射回 dim
        out_proj = self.out_proj(fused)  # [B, T, C]

        # reshape 回原图结构
        out = out_proj.transpose(1, 2).reshape(B, C, *spatial_dims)
        return out

# 通道和空间注意力机制
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca_weight = self.ca(x)
        x = x * ca_weight

        # Spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.sa(sa_input)
        x = x * sa_weight
        return x

