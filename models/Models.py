import torch.nn as nn
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature)


class DPCD(nn.Module):
    """ Change detection model

    Input :obj:`t1_img` and :obj:`t2_img`, extract encoder feature by :obj:`en_block1-4`,
    then exchange channel feature of :obj:`t1_feature` and :obj:`t2_feature`, and extract
    encoder feature by :obj:`en_block5`.
    输入：t1和t2的图片，通过en_block1-4提取编码器特征，随后交换t1_feature和t2_feature的通道特征，然后通过en_block5进一步提取编码器特征

    Upsample to get decoder feature by :obj:`de_block1-3`, get :obj:`seg_feature1` and :obj:`seg_feature2`
    by :obj:`seg_out1` and :obj:`seg_out2`.
    上采样：通过 de_bloack1-3获得编码器特征，

    Fuse t1 and t2 corresponding feature to get change feature by :obj:`dpfa` and :obj:`change_blcok`.
    通过dpfa和 change_block模块融合t1和t2的对应特征，生成变化特征

    Notice that output of module and model could be log in this model.

    Attribute:
        en_block(class): encoder feature extractor.
        channel_exchange(class): exchange t1 and t2 feature.
        de_block(class): decoder feature upsampler and extractor.
        dpfa(class): fuse t1 and t2 feature to get change feature by using both spatial and channel attention.
        change_block(class): change feature upsampler and extracor.
        seg_out(class): get decoder feature seg out result.
        upsample_x2(class): upsample change feature by 2.
        conv_out_change(class): conv out change feature out result.
    """

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
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3]) # 输入128通道，输出256通道
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4]) # 输入256通道，输出512通道

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
            nn.UpsamplingBilinear2d(scale_factor=2)  # 扩大输入特征图的空间尺寸（高度和宽度）至原来的2倍​​，但不会改变通道数
        )
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
            # T1_2和t2_2的格式均为 (B,64,H/2,W/2)
            t1_2 = self.en_block2(t1_1)
            t2_2 = self.en_block2(t2_1)

            # T1_3和t2_3的格式均为 (B,128,H/4,W/4)
            t1_3 = self.en_block3(t1_2)
            t2_3 = self.en_block3(t2_2)
            # t1_4和t2_4的格式均为(B,256,H/8,W/8)
            t1_4 = self.en_block4(t1_3)
            t2_4 = self.en_block4(t2_3)
            # 进过信道交换信息之后，t1_4和 t2_4的格式任然为(B,256,H/8,W/8)
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)
            # T1_5和t2_5的格式为(B,512,H/16,W/16)
            t1_5 = self.en_block5(t1_4)
            t2_5 = self.en_block5(t2_4)

        # 将 t1_5复制到del1_5，t2_5复制到del2_5
        # del1_5和del2_5的格式为(B,512,h/16,w/16)
        de1_5 = t1_5
        de2_5 = t2_5

        # 输入 ： 传入t1_4是为了将先前的256,1/8h,1/8w 和 对 512，h/16,w/16的de1_5经过上采样的结果。进行了上采样 -> contact - > 1*1的卷积
        # 对应 C-block2
        # 输出 ：（B,256,H/8,W/8）
        de1_4 = self.de_block1(de1_5, t1_4)
        de2_4 = self.de_block1(de2_5, t2_4)

        # 输入：传入t1_3是为了将先前的128,1/4h,1/4w 和 对 256，h/8,w/8的de1_4经过上采样的结果。进行了上采样 -> contact - > 1*1的卷积
        # 对应 C-block3
        # 输出： （B,128,H/4,W/4）
        de1_3 = self.de_block2(de1_4, t1_3)
        de2_3 = self.de_block2(de2_4, t2_3)

        # 输入：传入t1_2是为了将先前的64,1/2h,1/2w 和 对 128，h/4,w/4的de1_3经过上采样的结果。进行了上采样 -> contact - > 1*1的卷积
        # 对应 C-block4
        # 输出： （B,64,H/2,W/2）
        de1_2 = self.de_block3(de1_3, t1_2)
        de2_2 = self.de_block3(de2_3, t2_2)

        # 输入： （B,64,H/2,W/2）
        # 对应 C-HEAD模块
        # 输出：（B,1,H/2,W/2）
        seg_out1 = self.seg_out1(de1_2)
        seg_out2 = self.seg_out2(de2_2)

        if log:
            # 输入：del1_5 512通道，图片尺寸高 变为 w/16
            # 输出 ： 512通道，图片尺寸高 变为 w/16
            change_5 = self.dpfa1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_dpfa1',
                                  img_name=img_name)
            # 输入：change_5 (512,h/16,w/16) ;  de1_4_和de2_4(256,1/8h,1/8w) -> 在经过dpfa2（）得到(256,1/8h,1/8w)
            # 输出：进过 change_block4，输出通道数256，尺寸为1/8h,1/8w
            change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_dpfa2',
                                                               img_name=img_name))
            # 输入：change_4(256,1/8h,1/8w) ;de1_3和de2_3（128，h/4,w/4）进过 dpfa3()得到(128,h/4,w/4)
            # 输出：128，1/4h,1/4w
            change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_dpfa3',
                                                               img_name=img_name))
            # 输入：change_3(128，1/4h,1/4w);de1_2和de2_2(64,1/2h,1/2w)经过 dpfa4()得到 (64,1/2h,1/2w)
            # 输出： 64,1/2h,1/2w
            change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_dpfa4',
                                                               img_name=img_name))
        else:
            change_5 = self.dpfa1(de1_5, de2_5)

            change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4))

            change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3))

            change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2))
        # change的格式为(B,8,H,W)
        change = self.upsample_x2(change_2)
        # change_out的格式(B,1,H,W)
        change_out = self.conv_out_change(change)

        # 一般传递的关于log就是false,所以这个log_feature方法一般不会使用
        if log:
            log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model',
                        feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
                        img_name=img_name, module_output=False)

        return change_out, seg_out1, seg_out2
