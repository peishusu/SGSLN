from models.FHLCDNet_parts import *
from torchvision import models
from torchvision.models import VGG16_BN_Weights # torchvision 0.13+版本改变了加载预训练模型的方式


#增加decoder解码器，不concat多尺度特征生成guide map,也concat多尺度特征生成最后输出
class FHLCDNet(nn.Module):
    def __init__(self,):
        super(FHLCDNet, self).__init__()
        # vgg16_bn = models.vgg16_bn(pretrained=True)
        vgg16_bn = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1) # 新版本
        self.inc = vgg16_bn.features[:5]  # 通道数从 3 - >  64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        # TODO:这个模块有待替换 想替换成CTSR CTGP 这两个模块
        # self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1)
        # self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1)
        # self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1)
        # self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1)
        # fusion0 fusion1 对应CSTR模块
        self.fusion0 = XLSTM_axial(128, 128)
        self.fusion1 = XLSTM_axial(256,256)
        # fusion2 fusion3 对应 CTGP模块
        self.fusion2 = XLSTM_atten(512, 512)
        self.fusion3 = XLSTM_atten(512, 512)


        self.decoder = nn.Sequential(BasicConv2d(512,64,3,1,1),
                                     nn.Conv2d(64,1,3,1,1))

        self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1),
                                           nn.Conv2d(64, 1, 1))

        self.cgm_1 = Atten_Cross(128)
        self.cgm_2 = Atten_Cross(256)
        self.cgm_3 = Atten_Cross(512)
        self.cgm_4 = Atten_Cross(512)

        self.sa_1 = Atten_Spa(128)
        self.sa_2 = Atten_Spa(256)
        self.sa_3 = Atten_Spa(512)
        self.sa_4 = Atten_Spa(512)

        #相比v2 额外的模块
        self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024,512,3,1,1)
        self.decoder_module3 = BasicConv2d(768,256,3,1,1)
        self.decoder_module2 = BasicConv2d(384,128,3,1,1)

    def forward(self,A,B):
        '''
            我需要知道每层的输入而后输出合适
            输入的A(B,3,H,W) B(B,3,H,W)
        '''
        size = A.size()[2:]  # (H,W)
        layer1_pre = self.inc(A) # 输入格式 b,3,h,w 输出格式 b,64,h,w
        layer1_A = self.down1(layer1_pre) # 输入格式 b,64,h,w 输出格式 B,128,H/2,W/2
        layer2_A = self.down2(layer1_A) # 输入格式 b,128,h/2,w/2 输出格式 B,256,H/4,W/4
        layer3_A = self.down3(layer2_A) # 输入格式 b,256,h/4,w/4 输出格式 B,512,H/8,W/8
        layer4_A = self.down4(layer3_A) # 输入格式 b,512,h/8,w/8 输出格式 B,512,H/16,W/16

        layer1_pre = self.inc(B) # B,64,H,W
        layer1_B = self.down1(layer1_pre)  #B,128,H/2,W/2
        layer2_B = self.down2(layer1_B) # B,256,H/4,W/4
        layer3_B = self.down3(layer2_B) # B,512,H/8,W/8
        layer4_B = self.down4(layer3_B) # B,512,H/16,W/16

        # Concatenate features from A and B
        # TODO： 替换模块 为 CTSR 和 CTGP 模块  Done!!!!
        # layer1 = self.conv_reduce_1(torch.cat((layer1_B, layer1_A), dim=1)) # 输入格式B,128,H/2,W/2  输出格式： B,128,H/2,W/2
        # layer2 = self.conv_reduce_2(torch.cat((layer2_B, layer2_A), dim=1)) # 输入格式：B,256,H/4,W/4 输出格式：B,256,H/4,W/4
        # layer3 = self.conv_reduce_3(torch.cat((layer3_B, layer3_A), dim=1)) # 输入格式：B,512,H/8,W/8  输出格式： B，512，H/8,W/8
        # layer4 = self.conv_reduce_4(torch.cat((layer4_B, layer4_A), dim=1)) # 输入格式：B,512,H/16,W/16 输出格式：B，512,h/16,w/16
        # CTSR 模块
        # 第一层、第二层采样的图片进入 CSTR 模块
        layer1 = self.fusion0(layer1_A,layer1_B)  # 输入格式b,128,h/2,w/2 输出格式 b,128,h/2,w/2   fusion0 这个模块不改变b,c,h,w
        layer2 = self.fusion1(layer2_A,layer2_B)  # 输入格式b,256,h/4,w/4 输出格式 b,256,h/4,w/4   fusion1 这个模块不改变b,c,h,w
        # CTGP模块  第三层、第四层采样的图片进入 CTGP 模块
        layer3 = self.fusion2(layer3_A,layer3_B)  # 输入格式b,512,h/8,w/8 输出格式 b,512,h/8,w/8    fusion2 这个模块不改变b,c,h,w
        layer4 = self.fusion3(layer4_A,layer4_B)  # 输入格式b,512,h/16,w/16 输出格式 b,512,h/16,w/16  fusion3 这个模块不改变b,c,h,w



        # change semantic guiding map 这部分没用到
        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True) # layer4_1的格式为：B,512,H/2,W/2
        feature_fuse=layer4_1 #需要注释！
        change_map = self.decoder(feature_fuse)  #change_map 的格式为：B,1,H/2,W/2

        # Self_Attention模块
        layer4_4 = self.sa_4(layer4) # 输入格式为：B，512,h/16,w/16 输出格式：B,512,H/16,W/16
        layer4_5 = self.cgm_4(layer4_4,layer4) # Cross_Attention模块 # 输入格式为：layer4_4-> B,512,h/16,W/16 layer4 -> b,512,h/16,w/16  输出格式为：b,512,h/16,w/16
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4_5), layer3], 1)) # fushion module模块 输入格式：layer4_5->b,512,h/16,w/16 layer3->B,512,H/8,W/8 输出格式：B,512,h/8,w/8

        layer3_3 = self.sa_3(feature4) # Self_Attention模块 输入格式为：B,512,h/8,w/8 输出格式为:b,512,h/8,w/8
        layer3_4 = self.cgm_3(layer3_3,layer3) # cross attentio模块 输入格式为：layer3_3->b,512,h/8,w/8 layer3->B，512，H/8,W/8 输出格式:b,512,h/8,w/8
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3_4),layer2],1)) # fushion module模块 输入格式: layer3_4->b,512,h/8,w/8 layer2 -> B,256,H/4,W/4 输出格式:b,256,h/4,w/4

        layer2_3 = self.sa_2(feature3) # Self_Attention模块 输入格式为：b,256,h/4,w/4 输出格式为:b,256,h/4,w/4
        layer2_4 = self.cgm_2(layer2_3,layer2) # cross attentio模块 输入格式为：layer2_3->b,256,h/4,w/4 layer2->B,256,H/4,W/4 输出格式:b,256,h/4,w/4
        feature2 = self.decoder_module2(torch.cat([self.upsample2x(layer2_4), layer1], 1))  # fushion module模块 输入格式: layer2_4-b,256,h/4,w/4 layer1 ->  B,128,H/2,W/2 输出格式:b,128,h/2,w/2

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True) #输入格式 B,1,H/2,W/2 输出格式：b,1,h,w

        # 实际上就是Classfiler
        final_map = self.decoder_final(feature2) #输入格式 : B,128,H/2,W/2 输出格式: b,1,h/2,w/2
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True) # 输入格式 : B,1,H/2,W/2 输出格式: b,1,h,w
        # final_map 才是我们需要核心关注的输出
        return change_map, final_map


if __name__=='__main__':
    #测试热图
    net = FHLCDNet().cuda()
    out, final_map = net(torch.rand((2, 3, 256, 256)).cuda(), torch.rand((2, 3, 256, 256)).cuda())
    print(out.size()) # out的格式为 b,1,h,w
    print(final_map.size()) # final_map的格式为 b,1,h,w