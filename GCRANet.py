import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSA(nn.Module):
    # attention
    def __init__(self, in_channels):
        super().__init__()

        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, in_channels, 1)

        self.softmax = nn.Softmax(dim=1)  # 初始化一个Softmax操作

        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w

        key = self.conv_b(x).view(b * c, 1, n)
        query = self.conv_c(x).view(b * c, 1, n)
        value = self.conv_d(x)

        key = torch.nn.functional.normalize(key, dim=-1)
        query = torch.nn.functional.normalize(query, dim=-1)

        key = key.permute(0, 2, 1)
        kq = torch.bmm(query, key).view(b, c, 1, 1)

        #
        atten = self.softmax(kq*self.scale)
        feat_e = atten * value

        feat_e = self.conv_e(feat_e)

        return feat_e

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=[1], keepdim=True)
        var = x.var(dim=[1], keepdim=True)
        x = (x - mean) / (torch.sqrt(var + self.eps))
        return x * self.gamma + self.beta

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, 3, padding=1, stride=1, groups=dim * 4, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )

        self.dsa = DSA(dim)

    def forward(self, x):

        x = x + self.dsa(self.norm1(x))

        x = self.ffn(self.norm2(x)) + x

        return x

class CCAM(nn.Module):
    def __init__(self, channel1):
        super(CCAM, self).__init__()


        self.k = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=True,bn=False)
        self.q1 = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=True,bn=False)
        self.q2 = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=True,bn=False)
        self.pro = convbnrelu(channel1, channel1, k=1, s=1, p=0, relu=False,bn=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)

        self.scale = nn.Parameter(torch.ones(1))

        self.head_num = 4


    def forward(self, f_h, f_l, f_g):
        b, c, h, w = f_l.size()

        k = self.k(self.avg_pool(f_l)).view(b * self.head_num, c // self.head_num, 1)
        q1 = self.q1(self.avg_pool(f_g)).view(b * self.head_num, c // self.head_num, 1)
        q2 = self.q2(self.avg_pool(f_h)).view(b * self.head_num, c // self.head_num, 1)
        v = f_l.view(b * self.head_num, c // self.head_num, h * w)

        k = k.permute(0, 2, 1)
        atten1 = torch.bmm(q1, k)
        atten2 = torch.bmm(q2, k)
        atten = atten1 + atten2
        atten = self.softmax(atten * self.scale)
        out = torch.bmm(atten, v).view(b, c, h, w)
        out = self.pro(out)
        w_f_l = f_l + out

        return w_f_l

class FGAM(nn.Module):
    def __init__(self, channel1, channel2):
        super(FGAM, self).__init__()

        self.conv1 = convbnrelu(channel1, channel1, k=1, s=1, p=0, bn=False, relu=False)
        self.conv2 = convbnrelu(channel1, channel1, k=1, s=1, p=0, bn=False, relu=False)
        self.conv3 = convbnrelu(channel2, channel1, k=1, s=1, p=0, bn=False,relu=False)
        self.ram = CCAM(channel1)
        self.conv4 = convbnrelu(channel1,channel1, k=1, s=1, p=0,bn=False,relu=False)
        self.conv5 = convbnrelu(channel2,channel1, k=1, s=1, p=0,bn=False,relu=False)

    def forward(self, f_h, f_l, f_g):
        b, c, h, w = f_l.size()
        #
        f_h = self.conv1(f_h)
        f_l = self.conv2(f_l)
        w_f_l = self.ram(f_h, f_l,self.conv3(f_g))

        f_h = F.interpolate(f_h, size=(h, w), mode='bilinear', align_corners=True)
        f_h = self.conv4(f_h)

        f_g = self.conv5(f_g)
        f_g = F.interpolate(f_g, size=(h, w), mode='bilinear', align_corners=True)

        fused = w_f_l + f_h + f_g

        return fused

class GCRANet(nn.Module):
    def __init__(self, pretrained=None):
        super(GCRANet, self).__init__()
        self.context_path = VAMM_backbone(pretrained)

        # self.context_path.load_state_dict(torch.load("./SAMNet_backbone_pretrain.pth"))
        self.transformer = nn.Sequential(
            DSConv3x3(128, 128, stride=1),
            TransformerBlock(128),
            TransformerBlock(128),
        )

        self.prepare = convbnrelu(128,128, k=1, s=1, p=0, relu=False,bn=False)

        self.fgam1 = FGAM(96, 128)
        self.fgam2 = FGAM(64, 128)
        self.fgam3 = FGAM(32, 128)
        self.fgam4 = FGAM(16, 128)

        self.fuse = nn.ModuleList([
            DSConv3x3(128, 96, dilation=1),
            DSConv3x3(96, 64, dilation=2),
            DSConv3x3(64, 32, dilation=2),
            DSConv3x3(32, 16, dilation=2),
            DSConv3x3(16, 16, dilation=2)
        ])

        self.heads = nn.ModuleList([
            SalHead(in_channel=128),
            SalHead(in_channel=64),
            SalHead(in_channel=32),
            SalHead(in_channel=16),
            SalHead(in_channel=16),
        ])

    def forward(self, x):  # (3, 1)
        ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage5 = self.context_path(x)

        # #
        ct_stage6 = self.transformer(ct_stage5)

        fused_stage1 = self.fuse[0](self.prepare(ct_stage5))

        fused_stage2 = self.fuse[1](self.fgam1(fused_stage1, ct_stage4, ct_stage6))

        fused_stage3 = self.fuse[2](self.fgam2(fused_stage2, ct_stage3, ct_stage6))

        fused_stage4 = self.fuse[3](self.fgam3(fused_stage3, ct_stage2, ct_stage6))
        #
        fused_stage5 = self.fuse[4](self.fgam4(fused_stage4, ct_stage1, ct_stage6))


        output_side1 = interpolate(self.heads[0](ct_stage6), x.size()[2:])
        output_side2 = interpolate(self.heads[1](fused_stage2), x.size()[2:])
        output_side3 = interpolate(self.heads[2](fused_stage3), x.size()[2:])
        output_side4 = interpolate(self.heads[3](fused_stage4), x.size()[2:])
        output_main = self.heads[4](fused_stage5)

        return output_main, output_side2, output_side3, output_side4,output_side1



interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )

    def forward(self, x):
        return self.conv(x)




class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class VAMM_backbone(nn.Module):
    def __init__(self, pretrained=None):
        super(VAMM_backbone, self).__init__()
        self.layer1 = nn.Sequential(
            convbnrelu(3, 16, k=3, s=1, p=1),
            VAMM(16, dilation_level=[1, 2, 3])
        )
        self.layer2 = nn.Sequential(
            DSConv3x3(16, 32, stride=2),
            VAMM(32, dilation_level=[1, 2, 3])
        )
        self.layer3 = nn.Sequential(
            DSConv3x3(32, 64, stride=2),
            VAMM(64, dilation_level=[1, 2, 3]),
            VAMM(64, dilation_level=[1, 2, 3]),
            VAMM(64, dilation_level=[1, 2, 3])
        )
        self.layer4 = nn.Sequential(
            DSConv3x3(64, 96, stride=2),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3]),
            VAMM(96, dilation_level=[1, 2, 3])
        )
        self.layer5 = nn.Sequential(
            DSConv3x3(96, 128, stride=2),
            VAMM(128, dilation_level=[1, 2]),
            VAMM(128, dilation_level=[1, 2]),
            VAMM(128, dilation_level=[1, 2])
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out1, out2, out3, out4, out5


class VAMM(nn.Module):
    def __init__(self, channel, dilation_level=[1, 2, 4, 8], reduce_factor=4):
        super(VAMM, self).__init__()
        self.planes = channel
        self.dilation_level = dilation_level
        self.conv = DSConv3x3(channel, channel, stride=1)
        self.branches = nn.ModuleList([
            DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
        ])
        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = convbnrelu(channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.convs = nn.Sequential(
            convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
            DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=2),
            DSConv3x3(channel // reduce_factor, channel // reduce_factor, stride=1, dilation=4),
            nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        ### ChannelGate
        d = self.gap(gather)
        d = self.fc2(self.fc1(d))
        d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)

        ### SpatialGate
        s = self.convs(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)])) + x


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
from torchstat import stat
if __name__ == '__main__':

    model = GCRANet()
    from mmcv.cnn import get_model_complexity_info

    # if torch.cuda.is_available():
    #     net = model.cuda()
    # flops, params = get_model_complexity_info(net, input_shape=(3, 256, 256),print_per_layer_stat=True)
    # print(flops)
    # print(params)

    stat(model, (3, 256, 256))

    # from thop import profile, clever_format
    #
    # input = torch.randn(1, 3, 256, 256).cuda()
    # flops, params = profile(net, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.4f")
    # print(flops)
    # print(params)


