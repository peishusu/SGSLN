import torch
from torch import nn
from torch.cuda.amp import autocast
from enum import Enum
from models.CDXLSTM.vision_lstm import ViLBlock, SequenceTraversal
from torch.nn import functional as F
from functools import partial



# TODO：暂时存在这个想法，还尚未使用！！！复合型 LSTM模块
'''
复合型LSTM”通常是指结合了多种LSTM变体（如标准LSTM、mLSTM、sLSTM等）特点，融合多种门控机制和状态更新方式，形成一个更强大、更灵活的模型。它可能会：

    在同一层或多层中交替使用不同类型的LSTM单元
    
    将不同LSTM的输出进行融合（拼接、加权求和、门控融合等）
    
    在一个单元内部融合多种门控结构，形成新的计算流程
'''
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.pe = nn.Parameter(torch.zeros(1, channels, height, width))
        nn.init.normal_(self.pe, std=0.02)
    def forward(self, x):
        return x + self.pe

class CompositeLSTMLayerOptimized(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, num_layers=2, bidirectional=False, height=32, width=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.norm = nn.LayerNorm(input_dim)
        self.pos_enc = PositionalEncoding2D(input_dim, height, width)

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.W_m = nn.Linear(self.hidden_dim * (2 if bidirectional else 1), input_dim, bias=False)
        self.m_lstm_gates = nn.Linear(input_dim, 4 * self.hidden_dim)

        self.s_lstm_gates = nn.Linear(input_dim, 4 * self.hidden_dim)

        # 融合权重改为小型MLP动态融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * (2 if bidirectional else 1), 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )

        out_dim = self.hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, self.input_dim) if out_dim != self.input_dim else nn.Identity()

    @autocast()
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        spatial_dims = x.shape[2:]

        x = self.pos_enc(x)  # 添加位置编码

        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)  # [B, T, C]
        x_norm = self.norm(x_flat)

        lstm_out, _ = self.lstm(x_norm)  # [B, T, hidden_dim*out_dir]

        m = self.W_m(lstm_out) * x_norm
        gates = self.m_lstm_gates(m)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_m = f * 0 + i * g
        h_m = o * torch.tanh(c_m)

        gates_s = self.s_lstm_gates(x_norm)
        i_s, f_s, g_s, o_s = gates_s.chunk(4, dim=-1)
        i_s = torch.sigmoid(i_s)
        f_s = torch.sigmoid(f_s)
        g_s = torch.tanh(g_s)
        o_s = torch.sigmoid(o_s)
        c_s = f_s * 0 + i_s * g_s
        h_s = o_s * torch.tanh(c_s)

        # 动态计算融合权重，针对每个时间步
        fusion_w = self.fusion_mlp(lstm_out)  # [B, T, 3]
        fusion_w = fusion_w.unsqueeze(-1)  # [B, T, 3, 1]
        stacked = torch.stack([lstm_out, h_m, h_s], dim=2)  # [B, T, 3, hidden_dim]
        fused = (fusion_w * stacked).sum(dim=2)  # [B, T, hidden_dim]

        out_proj = self.proj(fused)
        out = out_proj.transpose(1, 2).reshape(B, C, *spatial_dims)

        # 残差连接
        out = out + x[:, :, :n_tokens].reshape(B, C, *spatial_dims)

        return out


#TODO: SK-Fusion LSTM / Selective Kernel LSTM（选择性核门控LSTM） 是一个非常适合遥感变化检测任务的替换方案
'''
在遥感变化检测任务中，尤其是复杂地物、尺度多样的场景下，变化模式具有显著的不确定性：

    有些区域发生了强变化（如建筑新建/拆除）
    
    有些区域变化很微弱（如水位线、农田）
    
    有些区域保持不变（如道路）
    
    所以我们需要模型能：
    
    动态决定哪些通道/特征要强调时序建模，哪些要弱化或绕开！
'''
class SKLSTMLayer(nn.Module):
    def __init__(self, dim, hidden_dim=None, reduction=8):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim

        # 分支1：标准 LSTM
        self.lstm1 = nn.LSTM(dim, self.hidden_dim, batch_first=True)

        # 分支2：模拟大感受野的 LSTM（可用 dilation 等替代）
        self.lstm2 = nn.LSTM(dim, self.hidden_dim, batch_first=True)

        # 分支3：identity 直通
        self.skip_proj = nn.Identity()

        # SK 融合门控
        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim // reduction)
        self.fc2 = nn.Linear(self.hidden_dim // reduction, 3)
        self.softmax = nn.Softmax(dim=1)

        # 输出映射回 dim
        self.out_proj = nn.Linear(self.hidden_dim, dim)

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

    @autocast(enabled=False)
    def forward(self, x):
        # 输入 x: [B, C, H, W] 或其他空间维度
        if x.dtype == torch.float16:
            x = x.float()

        B, C = x.shape[:2]
        assert C == self.dim
        spatial_dims = x.shape[2:]
        T = torch.tensor(spatial_dims).prod().item()

        # 展平空间维度： [B, C, H, W] -> [B, T, C]
        x_flat = x.reshape(B, C, T).transpose(1, 2)  # [B, T, C]
        x_norm = self.norm(x_flat)

        # 分支计算
        h1, _ = self.lstm1(x_norm)
        h2, _ = self.lstm2(x_norm)
        h3 = self.skip_proj(x_norm)

        # 融合分支
        feats = torch.stack([h1, h2, h3], dim=1)  # [B, 3, T, H]
        gap = feats.mean(dim=2).mean(dim=-1)     # [B, 3]
        gap_flat = torch.cat([gap[:, i] for i in range(3)], dim=-1)  # [B, H*3]

        attn = self.fc2(F.relu(self.fc1(gap_flat)))  # [B, 3]
        weights = self.softmax(attn).unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]

        out = (feats * weights).sum(dim=1)  # [B, T, H]
        out_proj = self.out_proj(out)       # [B, T, C]

        # 恢复为原始维度：[B, T, C] -> [B, C, H, W]
        out_final = out_proj.transpose(1, 2).reshape(B, C, *spatial_dims)
        return out_final


#TODO： 混和  +  选择性 LSTM模块去
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



class ViLLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim= self.dim,
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

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class LHBlock(nn.Module):
    def __init__(self, channels_l, channels_h):
        super().__init__()
        self.channels_l = channels_l
        self.channels_h = channels_h
        self.cross_size = 12
        self.cross_kv = nn.Sequential(
            nn.BatchNorm2d(channels_l),
            nn.AdaptiveMaxPool2d(output_size=(self.cross_size, self.cross_size)),
            nn.Conv2d(channels_l, 2 * channels_h, 1, 1, 0)
        )

        self.conv = conv_1x1(channels_l, channels_h)
        self.norm = nn.BatchNorm2d(channels_h)
        
        self.mlp_l = Mlp(in_features=channels_l, out_features=channels_l)
        self.mlp_h = Mlp(in_features=channels_h, out_features=channels_h)

    def _act_sn(self, x):
        _, _, H, W = x.shape
        inner_channel = self.cross_size * self.cross_size
        x = x.reshape([-1, inner_channel, H, W]) * (inner_channel**-0.5)
        x = F.softmax(x, dim=1)
        x = x.reshape([1, -1, H, W])
        return x
    
    def attn_h(self, x_h, cross_k, cross_v):
        B, _, H, W = x_h.shape
        x_h = self.norm(x_h)
        x_h = x_h.reshape([1, -1, H, W])  # n,c_in,h,w -> 1,n*c_in,h,w
        x_h = F.conv2d(x_h, cross_k, bias=None, stride=1, padding=0,
                        groups=B)  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
        x_h = self._act_sn(x_h)
        x_h = F.conv2d(x_h, cross_v, bias=None, stride=1, padding=0,
                        groups=B)  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
        x_h = x_h.reshape([-1, self.channels_h, H,
                        W])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)

        return x_h

    def forward(self, x_l, x_h):
        x_l = x_l + self.mlp_l(x_l)
        x_l_conv = self.conv(x_l)
        x_h = x_h + F.interpolate(x_l_conv, size=x_h.shape[2:], mode='bilinear')

        cross_kv = self.cross_kv(x_l)
        cross_k, cross_v = cross_kv.split(self.channels_h, 1)
        cross_k = cross_k.permute(0, 2, 3, 1).reshape([-1, self.channels_h, 1, 1])  # n*144,channels_h,1,1
        cross_v = cross_v.reshape([-1, self.cross_size * self.cross_size, 1, 1])  # n*channels_h,144,1,1

        x_h = x_h + self.attn_h(x_h, cross_k, cross_v) # [4, 40, 128, 128]
        x_h = x_h + self.mlp_h(x_h)

        return x_h


class CDXLSTM(nn.Module):
    def __init__(self, channels=[40, 80, 192, 384]):
        super().__init__()
        self.channels = channels
        # fusion0 fusion1 对应CSTR模块
        self.fusion0 = XLSTM_axial(channels[0], channels[0])
        self.fusion1 = XLSTM_axial(channels[1], channels[1])
        # fusion2 fusion3 对应 CTGP模块
        self.fusion2 = XLSTM_atten(channels[2], channels[2])
        self.fusion3 = XLSTM_atten(channels[3], channels[3])

        self.LHBlock1 = LHBlock(channels[1], channels[0])
        self.LHBlock2 = LHBlock(channels[2], channels[0])
        self.LHBlock3 = LHBlock(channels[3], channels[0])

        self.mlp1 = Mlp(in_features=channels[0], out_features=channels[0])
        self.mlp2 = Mlp(in_features=channels[0], out_features=2)
        self.dwc = dsconv_3x3(channels[0], channels[0])

    def forward(self, inputsA,inputsB):
        featuresA = inputsA # 这里面的featuresA, featuresB分别指的是下采样的四个阶段的图片
        featuresB = inputsB
        # CTSR 模块
        # 第一层、第二层采样的图片进入 CSTR 模块
        x_diff_0 = self.fusion0(featuresA[0], featuresB[0]) # 输入格式b,128,h/2,w/2 输出格式 b,128,h/2,w/2   fusion0 这个模块不改变b,c,h,w
        x_diff_1 = self.fusion1(featuresA[1], featuresB[1])  # 输入格式b,256,h/4,w/4 输出格式 b,256,h/4,w/4   fusion1 这个模块不改变b,c,h,w
        # CTGP模块  第三层、第四层采样的图片进入 CTGP 模块
        x_diff_2 = self.fusion2(featuresA[2], featuresB[2])  # 输入格式b,512,h/8,w/8 输出格式 b,512,h/8,w/8    fusion2 这个模块不改变b,c,h,w
        x_diff_3 = self.fusion3(featuresA[3], featuresB[3])# 输入格式b,512,h/16,w/16 输出格式 b,512,h/16,w/16  fusion3 这个模块不改变b,c,h,w

        # CSIF模块 TODO ：暂时不采用这个模块
        x_h = x_diff_0
        x_h = self.LHBlock1(x_diff_1, x_h) # 第一个CSIF模块 输入格式:x_h-> b,128,h/2,w/2  x_diff_1->b,256,h/4,w/4  输出格式:b,128,h/2,w/2
        x_h = self.LHBlock2(x_diff_2, x_h) # 第二个CSIF模块 输入格式:x_h->b,128,h/2,w/2  x_diff_2 -> b,512,h/8,w/8 输出格式为:b,128,h/2,w/2
        x_h = self.LHBlock3(x_diff_3, x_h) # 第三个CSIF模块 输入格式:x_h->b,128,h/2,w/2  x_diff_2 -> b,512,h/16,w/16 输出格式为:b,128,h/2,w/2


        # 最终的head模块
        out = self.mlp2(self.dwc(x_h) + self.mlp1(x_h))  # 输入格式为  b,128,h/2,w/2 中间形态 self.dwc(x_h) + self.mlp1(x_h) -> b,128,h/2,w/2 输出格式为:b,2,h/2,w/2

        # 这一块不知道干什么的 out 将
        out = F.interpolate(
            out,
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )
        return out
    
if __name__ == '__main__':
    net = CDXLSTM(channels = [128, 256, 512, 512]).cuda()
    x = [torch.randn(size=(4,128,128,128)).cuda(),  # b,128,h/2,w/2
         torch.randn(size=(4,256,64,64)).cuda(), # b,256,h/4,w/4
         torch.randn(size=(4,512,32,32)).cuda(), #b,512,h/8,w/8
         torch.randn(size=(4,512,16,16)).cuda()] # b,512,h/16/w/16
    print("开始测试了...")
    y = net([x,x])

    # print(y.shape)
    print(f"[🔥DEBUG] y shape: {y.shape}")
