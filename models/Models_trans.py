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
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
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
        # 经过这个dpfa的操作之后，输入和输出通道数是相同的，同时不改变图片尺寸大小
        self.dpfa1 = DPFA(in_channel=channel_list[0])
        self.dpfa2 = DPFA(in_channel=channel_list[0])
        self.dpfa3 = DPFA(in_channel=channel_list[0])
        self.dpfa4 = DPFA(in_channel=channel_list[0])

        # change path
        # the change block is the same as decoder block
        # the change block is used to fuse former and latter change features
        self.change_block4 = Decoder_Block(in_channel=channel_list[0], out_channel=channel_list[0])
        self.change_block3 = Decoder_Block(in_channel=channel_list[0], out_channel=channel_list[0])
        self.change_block2 = Decoder_Block(in_channel=channel_list[0], out_channel=channel_list[0])

        self.seg_out1 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3, stride=1, padding=1),
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
        self.pos_s8 = PA(256,32)
        self.pos_s4 = PA(128,32)
        self.pos_s2 = PA(64,32)


        # msca多尺度上下文聚合器的定义，这里面的size需要随时的根据测试的picture去进行更新
        self.CA_s16 = context_aggregator(in_chan=32, size=ph.patch_size//16)
        self.CA_s8 = context_aggregator(in_chan=32, size=ph.patch_size//8)
        self.CA_s4 = context_aggregator(in_chan=32, size=ph.patch_size//4)
        self.CA_s2 = context_aggregator(in_chan=32,size=ph.patch_size//2)

        # 通道数从64-》32,但是图片尺寸不变化
        self.conv_s8 = nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1)
        self.conv_s2 = nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1)


        # 通道数不变化，但是图片尺寸变为原来的两倍
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)


        # 经过分类器
        self.classifier1 = Classifier()
        self.classifier2 = Classifier()
        self.classifier3 = Classifier()



        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        self.dropout_en = nn.Dropout(p=0.3)  # 你可以根据实际调整 p


        # init parameters
        # using pytorch default init is enough
        # 把初始化打开！！有助于网络收敛更快、最终性能也更好
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
        # t1_1的格式 (B,32,H,W)
        t1_1 = self.en_block1(t1)
        # t2_1的格式 (B,32,H,W)
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
            # T1_2和t2_2的格式均为 (B,64,H/2,W/2)  used == out1_s4
            t1_2 = self.en_block2(t1_1)
            t2_2 = self.en_block2(t2_1)

            # T1_3和t2_3的格式均为 (B,128,H/4,W/4)  used == out1_s8
            t1_3 = self.en_block3(t1_2)
            t2_3 = self.en_block3(t2_2)

            # CE 暂时不采用，进过CE信息之后，t1_4和 t2_4的格式任然为(B,128,H/4,W/4)
            # t1_3, t2_3 = self.channel_exchange4(t1_3, t2_3)

            # t1_4和t2_4的格式均为(B,256,H/8,W/8)
            t1_4 = self.en_block4(t1_3)
            t2_4 = self.en_block4(t2_3)

            # T1_5和t2_5的格式为(B,512,H/16,W/16)
            t1_5 = self.en_block5(t1_4)
            t2_5 = self.en_block5(t2_4)
            # T1_5和t2_5的格式为(B,512,H/16,W/16)   used == out1_s16
            # t1_5 = self.upsample_sizex2(t1_5)
            # t2_5 = self.upsample_sizex2(t2_5)

        # 针对t1_2 t1_3 t1_5 进行通道数的转换,都转化成32通道数，但是图片尺寸不变化 预期是（B,32,h/32,/w/32）
        # out1_s16格式为(B,32,H/16,W/16)
        out1_s16 = self.pos_s16(t1_5)
        out2_s16 = self.pos_s16(t2_5)
        # out1_s8 格式为(B,32,H/8,W/8)
        out1_s8 = self.pos_s8(t1_4)
        out2_s8 = self.pos_s8(t2_4)
        # out1_s4 格式为(B,32,H/4,/W4)
        out1_s4 = self.pos_s4(t1_3)
        out2_s4 = self.pos_s4(t2_3)
        # out1_s2的格式为(B,32,H/2,W/2)
        out1_s2 = self.pos_s2(t1_2)
        out2_s2 = self.pos_s2(t2_2)

        # --------------------context aggregate (scale 16, scale 8, scale 4, scale 2)--------------------------
        # x1_s16[B,32,H/16,W/16]  x2_s16[B,32,H/16,W/16]
        x1_s16 = self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)


        # out1_s8格式为 [B,32,H/8,W/8]
        out1_s8 = self.conv_s8(torch.cat([self.upsamplex2(x1_s16), out1_s8], dim=1))  # 图片1
        out2_s8 = self.conv_s8(torch.cat([self.upsamplex2(x2_s16), out2_s8], dim=1))  # 图片2
        # x1_s8格式为[B,32,H/8,W/8]
        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)


        # out1_s4格式为[B,32,H/4,H/4]
        out1_s4 = self.conv_s4(torch.cat([self.upsamplex2(x1_s8), out1_s4], dim=1))
        out2_s4 = self.conv_s4(torch.cat([self.upsamplex2(x2_s8), out2_s4], dim=1))
        # x1_s4的格式为：[B,32,H/4,W/4]
        x1_s4 = self.CA_s4(out1_s4)
        x2_s4 = self.CA_s4(out2_s4)

        # out1_s2格式为[B,32,H/2,H/2]
        out1_s2 = self.conv_s2(torch.cat([self.upsamplex2(x1_s4), out1_s2], dim=1))
        out2_s2 = self.conv_s2(torch.cat([self.upsamplex2(x2_s4), out2_s2], dim=1))
        # x1_s2的格式为：[B,32,H/2,W/2]
        x1_s2 = self.CA_s2(out1_s2)
        x2_s2 = self.CA_s2(out2_s2)



        # 进行tfam操作
        # x1_s16格式为[B,32,H/16,W/16] 预期 change_s16的输出是(b,32,h/16,w/16)
        change_s16 = self.dpfa1(x1_s16,x2_s16)
        # 输入：x1_s8格式为[B,32,H/8,W/8] -> 所以 self.dpfa2的格式就是(B,32,H/8,W/8) change_s16的格式为(B,32,h/16,w/16) -> 最终change_s8的格式就是(B,32,H/8,W/8)
        change_s8 = self.change_block4(change_s16,self.dpfa2(x1_s8,x2_s8))
        # 输入：x1_s4格式为[B,32,H/4,W/4] -> 所以 self.dpfa2的格式就是(B,32,H/4,W/4) change_s8的格式为(B,32,h/8,w/8) -> 最终change_s4的格式就是(B,32,H/4,W/4)
        change_s4 = self.change_block3(change_s8,self.dpfa3(x1_s4,x2_s4))
        # 输入：x1_s2格式为[B,32,H/2,W/2] -> 所以 self.dpfa4的格式就是(B,32,H/2,W/2) change_s4的格式为(B,32,h/4,w/4) -> 最终change_s2的格式就是(B,32,H/2,W/2)
        change_s2 = self.change_block2(change_s4, self.dpfa4(x1_s2, x2_s2))

        # change的格式为(B,8,H,W)
        change = self.upsample_x2(change_s2)
        # change_out的格式(B,1,H,W)
        change_out = self.conv_out_change(change)

        # 加一个dropout正则话，防止过拟合
        # if self.training:
        #     change_out = self.dropout_en(change_out)

        # change_out的输出格式确实为(B,1,h,w)
        return change_out

