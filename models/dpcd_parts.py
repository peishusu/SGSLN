import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from utils.path_hyperparameter import ph
import cv2
from torchvision import transforms as T
from pathlib import Path
from einops import rearrange

#
class Conv_BN_ReLU(nn.Module):
    """
        Basic convolution.
        继承自 torch.nn.Module，是一个简单的卷积层结构，通常用于深度学习模型中进行图像特征提取
    """

    # 是 Python 类的构造函数，用来初始化类实例的属性。在这里，构造函数初始化了一个包含卷积层、批量归一化层和激活层的神经网络模块。
    def __init__(self, in_channel, out_channel, kernel, stride):
        # in_channel:输入通道数
        # out_channel：输出通道数
        # kernel:卷积核的大小
        # stride:步幅
        super().__init__() # 调用父类 nn.Module 的构造函数，初始化 PyTorch 模块

        # nn.Conv2d：二维卷积层，接受输入的 in_channel 通道，输出 out_channel 通道，卷积核大小为 kernel，步长为 stride，且使用 padding=kernel // 2，目的是保持输入和输出图像的大小一致（适用于奇数大小的卷积核）
            # 图像是二维的（高 × 宽），所以我们需要二维卷积层来处理图像的空间特征。
        # nn.BatchNorm2d：二维批量归一化层，通常用于深度网络中，帮助加速训练，改善收敛性，减轻内部协方差偏移。
        # nn.ReLU：ReLU 激活函数，增加非线性
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    # 前向传播， model(x) 时，PyTorch 会自动调用 forward 方法来计算模型的输出。实际上，model(x) 等价于 model.forward(x)，这两者都会通过 forward 方法来处理输入数据。
    # x表示输入的张量（输入图像）
    def forward(self, x):
        # conv_bn_relu 不是一个方法，它是一个 nn.Sequential 模块，包含了卷积层、批量归一化层和 ReLU 激活层。
        # nn.Sequential 是 PyTorch 中一个容器模块，用来顺序地组织多个层（如卷积、激活、池化等），使得你可以像操作单个模块一样按顺序执行多个操作。
        #
        output = self.conv_bn_relu(x)

        return output


class CGSU(nn.Module):
    """
        Basic convolution module.

    """
    # 用于定义这个模块的层级结构。在这里它的作用是初始化 CGSU 模块
    # in_channel：表示输入特征图的通道数
    def __init__(self, in_channel):
        super().__init__()

        mid_channel = in_channel // 2 # 将输入通道数分成两个部分，mid_channel 是 in_channel 的一半。这个分配会在后续的 channel_split 中用到。

        #
        self.conv1 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )
    # forward 方法只有在你将输入数据传递给 model 时才会执行
    # 举个例子：model = CGSU(in_channel=channel_list[0]) -> output = model(input_data)  input_data 就是你要传递给模型的输入数据。当你这样调用时，PyTorch 会自动调用 CGSU 类中的 forward 方法。
    def forward(self, x):
        # 将输入 x 按通道进行拆分
        x1, x2 = channel_split(x)
        # 将拆分后的 x1 输入到卷积层 conv1 中，经过卷积、批归一化、激活函数和 Dropout 处理后，得到处理后的 x1
        x1 = self.conv1(x1)
        # 将处理过的 x1 和未处理的 x2 按通道维度拼接（dim=1）。这意味着拼接后的输出特征图的通道数将是 x1 和 x2 的通道数之和，即 in_channel。
        output = torch.cat([x1, x2], dim=1)
        return output


class CGSU_DOWN(nn.Module):
    """
        Basic convolution module with stride=2.
        作用： CGSU_DOWN 是一个简单的卷积模块，通常用于网络中的下采样（downsampling）操作。其主要功能是减少特征图的空间尺寸（宽度和高度），以便提取更抽象的特征。
    """

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             stride=2, bias=False), # stride=2 会将输入特征图的尺寸缩小一半。
                                   nn.BatchNorm2d(in_channel), # 批归一化，用来规范化卷积操作后的输出，帮助加速训练并稳定训练过程。
                                   nn.ReLU(inplace=True), # ReLU 激活函数，用于对卷积结果进行非线性变换，帮助模型捕获更多的特征。
                                   nn.Dropout(p=ph.dropout_p) # Dropout 层用于在训练时随机丢弃一部分神经元，防止过拟合。
                                   ) # 最大池化层，核大小为 2x2，步幅为 2，也用于下采样。与卷积层不同，池化层是通过选择局部区域的最大值来压缩空间尺寸
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # remember the tensor should be contiguous
        # 将输入 x 传入第一个卷积层 self.conv1 中。由于该卷积层使用了 stride=2，它会将输入特征图的尺寸减小一半，并应用卷积操作、批归一化、激活函数和 Dropout。
        output1 = self.conv1(x)

        # respath
        # 将输入 x 传入最大池化层 self.conv_res 中。通过最大池化操作，输入特征图的尺寸也会减小一半，但池化操作不包括卷积过程。
        output2 = self.conv_res(x)

        # 最后，将 output1 和 output2 沿着通道维度（dim=1）进行拼接。output1 来自卷积操作，output2 来自池化操作。通过拼接，模型保留了卷积和池化提取的特征。
        # 拼接之后，通道数量 变为 之前的两倍
        output = torch.cat([output1, output2], dim=1)

        return output


class Changer_channel_exchange(nn.Module):
    """
        Exchange channels of two feature uniformly-spaced with 1:1 ratio.
        输入通道数 C 和 输出通道数 C 是一致的。
    """

    def __init__(self, p=2):
        super().__init__()
        self.p = p # 默认为 p=2，意味着每两个通道进行一次交换。

    def forward(self, x1, x2):
        N, C, H, W = x1.shape # N代表batch_size
        exchange_mask = torch.arange(C) % self.p == 0 # 每两个通道交换一次
        exchange_mask1 = exchange_mask.cuda().int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1

        return out_x1, out_x2


# double pooling fuse attention
class DPFA(nn.Module):
    """
        Fuse two feature into one feature.
        实际上就是：TFAM
    """

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # 通道方面
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        # torch,cat() b,c,4,1
        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        # 与 torch.cat（拼接）不同，stack() 会增加一个新的维度。
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # 空间方面
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        # 由于这两个张量的形状不同，特别是在通道数和高度/宽度上的差异，进行加法运算之前需要对这两个张量进行广播（broadcasting）以使它们的形状一致。
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        if log:
            log_list = [t1, t2, t1_spatial_attention, t2_spatial_attention, fuse]
            feature_name_list = ['t1', 't2', 't1_spatial_attention', 't2_spatial_attention', 'fuse']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return fuse


class CBAM(nn.Module):
    """
        Attention module.
        是一个注意力机制模块，用于提高神经网络模型的表现。CBAM 分为两部分：通道注意力模块（Channel Attention Module）和空间注意力模块（Spatial Attention Module）
    """

    def __init__(self, in_channel):
        '''
            作用： 类定义与初始化（__init__）
            输入：
                in_channel:输入的通道数
        '''
        super().__init__()
        self.k = kernel_size(in_channel) # 根据 in_channel 来计算卷积核的大小，目的是为通道注意力模块和空间注意力模块定义合适的卷积核大小
        # 通道注意力机制
        self.channel_conv = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2) # 1D 卷积层用于处理通道注意力。它接受 2 个输入通道并输出 1 个通道，卷积核的大小是通过 self.k 计算的，padding=self.k // 2 保证卷积操作后保持大小。
        self.avg_pooling = nn.AdaptiveAvgPool2d(1) # 自适应平均池化，输出大小为 (1, 1)。用于计算每个通道的平均值。
        self.max_pooling = nn.AdaptiveMaxPool2d(1) # 自适应最大池化，输出大小为 (1, 1)。用于计算每个通道的最大值。
        self.sigmoid = nn.Sigmoid() # Sigmoid 激活函数，用于将计算出的注意力值限制在 [0, 1] 范围内

        # 空间注意力机制
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x, log=False, module_name=None, img_name=None):
        '''
           处理了输入特征图 x，并通过通道注意力和空间注意力机制对特征图进行了加权，最终生成经过注意力调整的输出。、
           输入x :（batch,channel,h,w）
        '''
        # 通道注意力模块:针对每个通道内的 所有像素进行操作
        # avg_pooling -> (b,c,1,1) squeeze -> (b,c,1) transpose -> (b,1,c)
        avg_channel = self.avg_pooling(x).squeeze(-1).transpose(1, 2)  # batch,1,channel
        max_channel = self.max_pooling(x).squeeze(-1).transpose(1, 2)  # batch,1,channel
        channel_weight = self.channel_conv(torch.cat([avg_channel, max_channel], dim=1))
        channel_weight = self.sigmoid(channel_weight).transpose(1, 2).unsqueeze(-1)  # b,c,1,1
        # 将计算出的通道注意力权重 channel_weight 与输入图像 x 相乘。通过这种方式，重要的通道（权重大）会被增强，不重要的通道（权重小）会被抑制。
        x = channel_weight * x

        # 空间注意力模块：针对每个空间位置上面的 所有像素进行操作
        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # b,1,h,w，对输入 x 的每个空间位置（每个像素）沿通道维度进行平均池化
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]  # b,1,h,w，对每个像素位置的通道进行最大池化
        # torch.cat([avg_spatial, max_spatial], dim=1) 这个操作会形成[b,2c,h,w]
        spatial_weight = self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))  # b,1,h,w
        # 使用 Sigmoid 激活函数，将空间权重压缩到 [0, 1] 范围
        spatial_weight = self.sigmoid(spatial_weight)
        # 将空间注意力权重 spatial_weight 与输入图像 x 相乘。这样，空间中重要的区域会被增强，不重要的区域会被抑制。
        output = spatial_weight * x

        if log:
            log_list = [spatial_weight]
            feature_name_list = ['spatial_weight']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return output


class Encoder_Block(nn.Module):
    """ Basic block in encoder"""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            CGSU_DOWN(in_channel=in_channel),
            CGSU(in_channel=out_channel),
            CGSU(in_channel=out_channel)
        )
        self.conv3 = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
        self.cbam = CBAM(in_channel=out_channel)

    def forward(self, x, log=False, module_name=None, img_name=None):
        x = self.conv1(x)
        x = self.conv3(x)
        x_res = x.clone()
        if log:
            output = self.cbam(x, log=log, module_name=module_name + '-x_cbam', img_name=img_name)
        else:
            output = self.cbam(x)
        output = x_res + output

        return output


class Decoder_Block(nn.Module):
    """
        Basic block in decoder.
        将
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()

        # assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest') # 仅改变输入特征图的空间分辨率（高度和宽度）变为原来的两倍，不改变通道数
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de) # de就变成b,c 2h,2w
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


'''
    改变输入和输出的通道数的,不改变尺寸
'''
class PA(nn.Module):
    def __init__(self, inchan = 512, out_chan = 32):
        super().__init__()
        self.conv = nn.Conv2d(inchan, out_chan, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.re = nn.ReLU()
        self.do = nn.Dropout2d(0.2)

        self.pa_conv = nn.Conv2d(out_chan, out_chan, kernel_size=1, padding=0, groups=out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv(x)
        x = self.do(self.re(self.bn(x0)))
        return x0 *self.sigmoid(self.pa_conv(x))



class context_aggregator(nn.Module):
    def __init__(self, in_chan=32, size=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = 32, size = size, heads = 8)

    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out

'''
    是一个 基于 Transformer 的视觉 Token 编码器，用于将空间特征图转换为全局上下文 Token
'''
class token_encoder(nn.Module):
    '''
        params
            in_chan	输入特征通道数（默认 32）。
            token_len	Token 数量（默认 4），控制上下文信息的压缩程度。
            heads	Transformer 多头注意力头数（默认 8）。
    '''
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        # 通过 1×1 卷积生成 token_len 个空间注意力图。
        # 输入：B C H W  输出 B token_len H W
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        # 位置编码：为 Token 添加可学习的位置信息（类似 ViT 的 [CLS] Token）
        # pos_embedding 形状：[1, token_len, C]，广播机制自动将[]
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        # transformer编码器，dim就是输入的通道数
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x) ## [B, token_len, H, W]
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous() # [B, token_len, H*W]
        spatial_attention = torch.softmax(spatial_attention, dim=-1) # 归一化，输入和输出一致
        # 特征图展平
        x = x.view([b, c, -1]).contiguous() # B C H*W

        # einsum运算
        # 输入 spatial_attention[B, token_len, H*W]  x[ B C H*W]  输出：[B, token_len, C]
        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)
        # pos_embedding 形状：[1, token_len, C]，广播机制自动将[B,token_len,C]
        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, size = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        # x指的是feature
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x

'''
    transformer编码器
'''
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

'''
    MHA模块
'''
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 输入x B tolen_len C    输出3 × [B, token_len, C]
        qkv = self.to_qkv(x).chunk(3, dim = -1) #
        # Q K V 每个都是[B, h, token_len, C]
        # 实际上就是对应 MHA Blocks片段执行流程
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn运算
        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # 最终输出格式：[B,token_len,C]
        return out


'''
    FFN模块
'''
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), # 特征扩展	[B, N, D] → [B, N, hidden_dim]
            nn.GELU(), #  激活函数	形状不变 [B, N, hidden_dim]
            nn.Dropout(dropout), # 随机失活	形状不变 [B, N, hidden_dim]
            nn.Linear(hidden_dim, dim), # 特征压缩	[B, N, hidden_dim] → [B, N, D]
            nn.Dropout(dropout) # 	随机失活	形状不变 [B, N, D]
        )
    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            # 除了通道数从 64->32 ，其他就是H W 不变
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan), # 对输入数据进行批归一化，以提高训练的稳定性和速度。
                            nn.ReLU(), # ReLU 激活函数加入非线性，帮助网络学习复杂的特征。
                            # 除了通道数从 32->2 ，其他就是H W 不变
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1)) #
    def forward(self, x):
        x = self.head(x)
        return x



# from ECANet, in which y and b is set default to 2 and 1
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


def channel_split(x):
    """Half segment one feature on channel dimension into two features, mixture them on channel dimension,
    and split them into two features."""

    # 获取输入 x 的形状：batchsize, num_channels, height, width
    batchsize, num_channels, height, width = x.data.size()
    # # 确保通道数是 4 的倍数，以便可以平分
    assert (num_channels % 4 == 0)
    #
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    # 最终输出的结果:batchsize, num_channels // 2, height, width
    return x[0], x[1]


def log_feature(log_list, module_name, feature_name_list, img_name, module_output=True):
    """ Log output feature of module and model
        作用是记录（log）神经网络模块的输出特征，并将其保存为图像文件。
    Log some output features in a module. Feature in :obj:`log_list` should have corresponding name
    in :obj:`feature_name_list`.

    For module output feature, interpolate it to :math:`ph.patch_size`×:math:`ph.patch_size`,
    log it in :obj:`cv2.COLORMAP_JET` format without other change,
    and log it in :obj:`cv2.COLORMAP_JET` format with equalization.
    For model output feature, log it without any change.

    Notice that feature is log in :obj:`ph.log_path`/:obj:`module_name`/
    name in :obj:`feature_name_list`/:obj:`img_name`.jpg.

    Parameter:
        log_list(list): list of output need to log.
        module_name(str): name of module which output the feature we log,
            if :obj:`module_output` is False, :obj:`module_name` equals to `model`.
        module_output(bool): determine whether output is from module or model.
        feature_name_list(list): name of output feature.
        img_name(str): name of corresponding image to output.


    """
    for k, log in enumerate(log_list):
        log = log.clone().detach()
        b, c, h, w = log.size()
        if module_output:
            log = torch.mean(log, dim=1, keepdim=True)
            log = F.interpolate(
                log * 255, scale_factor=ph.patch_size // h,
                mode='nearest').reshape(b, ph.patch_size, ph.patch_size, 1) \
                .cpu().numpy().astype(np.uint8)
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_equalize_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '_equalize/'
            Path(log_equalize_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                log_i = cv2.applyColorMap(log[i], cv2.COLORMAP_JET)
                cv2.imwrite(log_dir + img_name[i] + '.jpg', log_i)

                log_i_equalize = cv2.equalizeHist(log[i])
                log_i_equalize = cv2.applyColorMap(log_i_equalize, cv2.COLORMAP_JET)
                cv2.imwrite(log_equalize_dir + img_name[i] + '.jpg', log_i_equalize)
        else:
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log = torch.round(torch.sigmoid(log))
            log = F.interpolate(log, scale_factor=ph.patch_size // h,
                                mode='nearest').cpu()
            to_pil_img = T.ToPILImage(mode=None)
            for i in range(b):
                log_i = to_pil_img(log[i])
                log_i.save(log_dir + img_name[i] + '.jpg')
