import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature,PA,context_aggregator,Classifier)
from utils.path_hyperparameter import ph

class DPCD(nn.Module):

    def __init__(self):
        super().__init__()
        # 通道数量
        channel_list = [32, 64, 128, 256, 512]
        # 编码器部分
        # 这个encode1作用：
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=2),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       )
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3]) #
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4]) #

        self.channel_exchange4 = Changer_channel_exchange()

        # decoder：解码器部分
        # c-block2
        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        # c-block3
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        # c-block4
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        # dpfa(可以认为就是论文中tfam的实现)
        # 经过这个dpfa的操作之后，输入和输出通道数是相同的
        self.dpfa1 = DPFA(in_channel=channel_list[4])
        self.dpfa2 = DPFA(in_channel=channel_list[3])
        self.dpfa3 = DPFA(in_channel=channel_list[2])
        self.dpfa4 = DPFA(in_channel=channel_list[1])

        # change path
        # the change block is the same as decoder block
        # the change block is used to fuse former and latter change features
        self.change_block4 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.change_block3 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.change_block2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.seg_out1 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)  # 扩大输入特征图的空间尺寸（高度和宽度）至原来的2倍，但不会改变通道数
        )

        # 作用，通道数不变，将H,W扩大到原来的2倍
        self.upsample_sizex2 = nn.ConvTranspose2d(
            in_channels=channel_list[4],
            out_channels=channel_list[4],
            kernel_size=3,
            stride=2,  # 步长2实现2倍上采样
            padding=1,
            output_padding=1  # 确保尺寸精确加倍
        )

        self.pos_s16 = PA(512,32)
        self.pos_s8 = PA(128,32)
        self.pos_s4 = PA(64,32)


        # msca多尺度上下文聚合器的定义
        self.CA_s16 = context_aggregator(in_chan=32, size=ph.patch_size//16)
        self.CA_s8 = context_aggregator(in_chan=32, size=ph.patch_size//8)
        self.CA_s4 = context_aggregator(in_chan=32, size=ph.patch_size//4)

        self.conv_s8 = nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)


        # 经过分类器
        self.classifier1 = Classifier()
        self.classifier2 = Classifier()
        self.classifier3 = Classifier()



        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)


        # init parameters
        # using pytorch default init is enough
        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, t1, t2, log=False, img_name=None):
        '''
            params：
                t1(B,3,H,W)
                t2(B,3,H,W)
        '''
        # t1_1的格式 (B,32,H/2,/2)
        t1_1 = self.en_block1(t1)
        # t2_1的格式 (B,32,H/2,W/2)
        t2_1 = self.en_block1(t2)

        if log:
            # 输入32通道，输出 64通道，同时 图片尺寸高 变为 w/2
            t1_2 = self.en_block2(t1_1, log=log, module_name='t1_1_en_block2', img_name=img_name)
            t2_2 = self.en_block2(t2_1, log=log, module_name='t2_1_en_block2', img_name=img_name)

            # 输入64通道，输出128通道，同时 图片尺寸高 变为 w/4
            t1_3 = self.en_block3(t1_2, log=log, module_name='t1_2_en_block3', img_name=img_name)
            t2_3 = self.en_block3(t2_2, log=log, module_name='t2_2en_block3', img_name=img_name)

            # 输入 128通道，输出 256通道，图片尺寸高 变为 w/8
            t1_4 = self.en_block4(t1_3, log=log, module_name='t1_3_en_block4', img_name=img_name)
            t2_4 = self.en_block4(t2_3, log=log, module_name='t2_3_en_block4', img_name=img_name)

            # 进入CE环节，实际上就是将t1和t2的图片通道进行了交换，但是实际上每个图片的通道数都没变化，同时图片的尺寸大小任然是 1/8
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

            # 输入 256通道，输出 512通道，图片尺寸高 变为 w/16
            t1_5 = self.en_block5(t1_4, log=log, module_name='t1_4_en_block5', img_name=img_name)
            t2_5 = self.en_block5(t2_4, log=log, module_name='t2_4_en_block5', img_name=img_name)
        else:
            # T1_2和t2_2的格式均为 (B,64,H/4,W/4)  used == out1_s4
            t1_2 = self.en_block2(t1_1)
            t2_2 = self.en_block2(t2_1)

            # T1_3和t2_3的格式均为 (B,128,H/8,W/8)  used == out1_s8
            t1_3 = self.en_block3(t1_2)
            t2_3 = self.en_block3(t2_2)

            # t1_4和t2_4的格式均为(B,256,H/16,W/16)
            t1_4 = self.en_block4(t1_3)
            t2_4 = self.en_block4(t2_3)

            # 进过CE信息之后，t1_4和 t2_4的格式任然为(B,256,H/16,W/16)
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

            # T1_5和t2_5的格式为(B,512,H/32,W/32)
            t1_5 = self.en_block5(t1_4)
            t2_5 = self.en_block5(t2_4)
            # T1_5和t2_5的格式为(B,512,H/16,W/16)   used == out1_s16
            t1_5 = self.upsample_sizex2(t1_5)
            t2_5 = self.upsample_sizex2(t2_5)

        # 针对t1_2 t1_3 t1_5 进行通道数的转换,都转化成32通道数，但是图片尺寸不变化 预期是（B,32,h/16/w/16）
        out1_s16 = self.pos_s16(t1_5)
        out2_s16 = self.pos_s16(t2_5)
        # out1_s8 格式为(B,32,H/8,W/8)
        out1_s8 = self.pos_s8(t1_3)
        out2_s8 = self.pos_s8(t2_3)
        # out1_s4 格式为(B,32,H/4,/W4)
        out1_s4 = self.pos_s4(t1_2)
        out2_s4 = self.pos_s4(t2_2)

        # --------------------context aggregate (scale 16, scale 8, scale 4)--------------------------
        x1_s16 = self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)
        # 输入 x1_s16[B,32,H/16,W/16]  x2_s16[B,32,H/16,W/16]   输出：x16[B,64,H/16,W/16]
        x16 = torch.cat([x1_s16, x2_s16], dim=1)
        # 指定了输出图像的目标大小，它等于原始图像的高度和宽度 (H, W)。
        # 输入[B,64,H/16,W/16]  输出格式：(B, 64, H, W)
        x16 = F.interpolate(x16, size=t1.shape[2:], mode='bicubic', align_corners=True)
        # x16的格式变为 [B,2,H,W]
        x16 = self.classifier1(x16)

        # out1_s8格式为 [B,32,H/8,W/8]
        out1_s8 = self.conv_s8(torch.cat([self.upsamplex2(x1_s16), out1_s8], dim=1))  # 图片1
        out2_s8 = self.conv_s8(torch.cat([self.upsamplex2(x2_s16), out2_s8], dim=1))  # 图片2
        # x1_s8格式为[B,32,H/8,W/8]
        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)
        # x8的最终输出格式[B,2,H,W]
        x8 = torch.cat([x1_s8, x2_s8], dim=1)
        x8 = F.interpolate(x8, size=t1.shape[2:], mode='bicubic', align_corners=True)
        x8 = self.classifier2(x8)

        # out1_s4格式为[B,32,H/4,H/W]
        out1_s4 = self.conv_s4(torch.cat([self.upsamplex2(x1_s8), out1_s4], dim=1))
        out2_s4 = self.conv_s4(torch.cat([self.upsamplex2(x2_s8), out2_s4], dim=1))
        x1 = self.CA_s4(out1_s4)
        x2 = self.CA_s4(out2_s4)
        # x的最终输出格式:[B,2,H,W]
        x = torch.cat([x1, x2], dim=1)
        x = F.interpolate(x, size=t1.shape[2:], mode='bicubic', align_corners=True)
        x = self.classifier3(x)





        # 一般传递的关于log就是false,所以这个log_feature方法一般不会使用
        # if log:
        #     log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model',
        #                 feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
        #                 img_name=img_name, module_output=False)

        return x, x8, x16

        # return change_out, seg_out1, seg_out2
