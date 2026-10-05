# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from toolbox.models.torch_vertex import Graphers 
import torchvision.models as models

from thop import profile
from thop import clever_format

class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x
    
class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(decoder, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 1, padding=0)
        self.conv2 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 1, padding=0)
        self.conv3 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 3)
        self.conv4 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 3)
        self.conv5 = ConvBnLeakyRelu2d(in_channels * 2, out_channels, kernel_size = 3)
        
    def forward(self, x, y):
        
        x1 = self.conv1(x)
        y1 = self.conv2(y)
        x2 = x1 + y1
        y2 = x1 * y1
        x1 = self.conv3(x2)
        y1 = self.conv4(y2)
        x2 = x1 + x
        y2 = y1 + y
        
        return self.conv5(torch.cat((x2, y2), dim = 1))
    
class ori_classifier(nn.Module):
    def __init__(self, in_channels, num_class, scale, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ori_classifier, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 1, padding=0)
        self.conv2 = ConvBnLeakyRelu2d(in_channels, num_class, kernel_size = 3)
        self.scale = scale
        
    def forward(self, x):
        
        x = self.conv1(x) + x
        x = F.upsample(x, scale_factor=self.scale, mode='nearest')
        x = self.conv2(x)
        
        return x
    
class classifier(nn.Module):
    def __init__(self, in_channels, num_class, scale, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(classifier, self).__init__()
        self.conv1 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 1, padding=0)
        self.conv2 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 3)
        self.conv3 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 3)
        self.conv4 = ConvBnLeakyRelu2d(in_channels, in_channels, kernel_size = 3)
        
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding=1)
        self.conv6 = nn.Conv2d(in_channels, num_class*2, kernel_size = 3, padding=1)
        self.conv7 = nn.Conv2d(num_class*2, num_class, kernel_size = 3, padding=1)
        self.conv8 = nn.Conv2d(in_channels, num_class*2, kernel_size = 3, padding=1)
        self.scale = scale
        self.drop  = nn.Dropout2d(p = 0.01) 
        
    def forward(self, x):
        ori_x = x
        x = self.drop(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + ori_x
        x = self.conv3(x)
        x = F.upsample(x, scale_factor=self.scale, mode='nearest')
        x = self.conv4(x) + x
        x = self.conv5(x) + x
        x = self.conv6(x) + self.conv8(F.upsample(ori_x, scale_factor=self.scale, mode='nearest'))
        return self.conv7(x)
 


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class Multi_Concat_Block(nn.Module):
    def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
        super(Multi_Concat_Block, self).__init__()
        c_ = int(c2 * e)
        
        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList(
            [Conv(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
        )
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        
        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
            
        out = self.cv4(torch.cat([x_all[id] for id in self.ids], 1))
        return out

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
    
class Transition_Block(nn.Module):
    def __init__(self, c1, c2):
        super(Transition_Block, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        
        self.mp  = MP()

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        
        return torch.cat([x_2, x_1], 1)

class DGRFNet(nn.Module):

    def __init__(self, n_classes):
        super(DGRFNet, self).__init__()
        
        model1 = models.resnet18(pretrained=True) # 18 34 50 152
        model2 = models.resnet18(pretrained=True) # 18 34 50 152
        
        # RGB encoding via resnet152 
        self.encoder_RGB3_conv1 = model1.conv1
        self.encoder_RGB3_bn1 = model1.bn1
        self.encoder_RGB3_relu = model1.relu
        self.encoder_RGB3_maxpool = model1.maxpool
        self.encoder_RGB3_layer1 = model1.layer1
        self.encoder_RGB3_layer2 = model1.layer2
        self.encoder_RGB3_layer3 = model1.layer3
        self.encoder_RGB3_layer4 = model1.layer4
        
        # T encoding via resnet152
        self.encoder_T3_conv1 = model2.conv1
        self.encoder_T3_bn1 = model2.bn1
        self.encoder_T3_relu = model2.relu
        self.encoder_T3_maxpool = model2.maxpool
        self.encoder_T3_layer1 = model2.layer1
        self.encoder_T3_layer2 = model2.layer2
        self.encoder_T3_layer3 = model2.layer3
        self.encoder_T3_layer4 = model2.layer4

        
        k = 9
        act = 'gelu'
        norm = 'batch'
        bias = True
        epsilon = 0.2
        stochastic = False
        conv = 'mr'
        drop_path = 0.01
        blocks = [1] # ori: [2,2,18,2]
        self.n_blocks = sum(blocks)
        # channels = [128]# channels = [128, 256, 512, 1024]
        reduce_ratios = [1, 2, 4, 8] # 4, 2, 1, 1
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        idx = 0
        # self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 30, 40))
        
        self.GCN0 = Graphers(512, 512, num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                      bias, stochastic, epsilon, reduce_ratios[0], n=15*20, drop_path=dpr[idx],
                      relative_pos=False) # relative_pos=True
        self.GCN1 = Graphers(256, 256, num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                      bias, stochastic, epsilon, reduce_ratios[1], n=30*40, drop_path=dpr[idx],
                      relative_pos=False) # relative_pos=True
        self.GCN2 = Graphers(128, 128, num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                      bias, stochastic, epsilon, reduce_ratios[2], n=60*80, drop_path=dpr[idx],
                      relative_pos=False) # relative_pos=True
        self.GCN3 = Graphers(64, 64, num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                      bias, stochastic, epsilon, reduce_ratios[3], n=120*160, drop_path=dpr[idx],
                      relative_pos=False) # relative_pos=True
        
        
        transition_channels         = 32
        panet_channels              = 8 
        e                           = 2
        n                           = 4
        ids                         = [-1, -2, -3, -4, -5, -6]
        self.upsample               = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_for_FP            = Conv(transition_channels * 16, transition_channels * 16)
        self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2         = Conv(transition_channels * 8, transition_channels * 8)
        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1         = Conv(transition_channels * 4, transition_channels * 4)
        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        self.conv_for_P3            = Conv(transition_channels * 4, transition_channels * 2)
        self.conv_for_feat0         = Conv(transition_channels * 2, transition_channels * 2)
        self.conv3_for_upsample3    = Multi_Concat_Block(transition_channels * 4, panet_channels * 2, transition_channels * 2, e=e, n=n, ids=ids)

        self.down_sample0           = Transition_Block(transition_channels * 2, transition_channels * 2)
        self.conv3_for_downsample0  = Multi_Concat_Block(transition_channels * 8, panet_channels * 4, transition_channels * 4, e=e, n=n, ids=ids)
        

        self.down_sample1           = Transition_Block(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2           = Transition_Block(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)

        self.rep_conv_0 = Conv(transition_channels * 2, transition_channels * 2, 3, 1)        
        self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 4, 3, 1)
        self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 8, 3, 1)
        self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 16, 3, 1)
        
        
        self.decoder0    = decoder(512, 256)
        self.classifier0 = ori_classifier(256, 2, 16)
        self.decoder1    = decoder(256, 128)
        self.classifier1 = ori_classifier(128, 2, 8)
        self.decoder2    = decoder(128, 64)
        self.classifier2 = ori_classifier(64, n_classes, 4)
        self.decoder3    = decoder(64, 32)
        self.classifier3 = classifier(32, n_classes, 2)
        
    def forward(self, rgb, depth):
        # split data into RGB and INF
        x_rgb = rgb
        ir = depth[:, :1, ...]
        x_inf = torch.cat((ir, ir, ir), dim=1)

        # encode
        rgb = self.encoder_RGB3_conv1(x_rgb)
        rgb = self.encoder_RGB3_bn1(rgb)
        rgb = self.encoder_RGB3_relu(rgb)
        rgb = self.encoder_RGB3_maxpool(rgb)           
        rgb = self.encoder_RGB3_layer1(rgb)  
        fr3 = rgb #1/4
        rgb = self.encoder_RGB3_layer2(rgb)
        fr2 = rgb #1/8
        rgb = self.encoder_RGB3_layer3(rgb)
        fr1 = rgb #1/16
        rgb = self.encoder_RGB3_layer4(rgb)
        fr0 = rgb #1/32

        tt = self.encoder_T3_conv1(x_inf)
        tt = self.encoder_T3_bn1(tt)
        tt = self.encoder_T3_relu(tt)
        tt = self.encoder_T3_maxpool(tt)
        tt = self.encoder_T3_layer1(tt)
        ft3 = tt #1/4
        tt = self.encoder_T3_layer2(tt)
        ft2 = tt #1/8
        tt = self.encoder_T3_layer3(tt)
        ft1 = tt #1/16
        tt = self.encoder_T3_layer4(tt)
        ft0 = tt #1/32
        
        # fusion & decode
        ff0 = self.GCN0(fr0, ft0) # torch.Size([1, 512, 15, 20])
        ff1 = self.GCN1(fr1, ft1) # torch.Size([1, 256, 30, 40])
        ff2 = self.GCN2(fr2, ft2) # torch.Size([1, 128, 60, 80])
        ff3 = self.GCN3(fr3, ft3) # torch.Size([1, 64, 120, 160])
        
        # feature enhancement
        P5 = self.conv_for_FP(ff0) # 512, 15, 20
        P5_conv = self.conv_for_P5(P5) # 512, 15, 20 --> 256, 15, 20
        P5_upsample = self.upsample(P5_conv) # 256, 30, 40
        P4          = torch.cat([self.conv_for_feat2(ff1), P5_upsample], 1)
        P4          = self.conv3_for_upsample1(P4) # 256, 30, 40    
        
        P4_conv     = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3          = torch.cat([self.conv_for_feat1(ff2), P4_upsample], 1)
        P3          = self.conv3_for_upsample2(P3) # 128, 60, 80
        
        P3_conv     = self.conv_for_P3(P3)
        P3_upsample = self.upsample(P3_conv)
        P2          = torch.cat([self.conv_for_feat0(ff3), P3_upsample], 1)
        P2          = self.conv3_for_upsample3(P2) # 64, 120, 160
        
        P2_downsample = self.down_sample0(P2)
        P3 = torch.cat([P2_downsample, P3], 1)
        P3 = self.conv3_for_downsample0(P3)
        
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        
        ff3 = self.rep_conv_0(P2)
        ff2 = self.rep_conv_1(P3)
        ff1 = self.rep_conv_2(P4)
        ff0 = self.rep_conv_3(P5)
        
        
        # decode & classify
        x = self.decoder0(ff0, ff0)
        x = F.upsample(x, scale_factor=2, mode='nearest') # stage0
        sal = self.classifier0(x) # torch.Size([2, 9, 60, 80])
        
        x = self.decoder1(x, ff1)
        x = F.upsample(x, scale_factor=2, mode='nearest') # stage1
        edge = self.classifier1(x) # torch.Size([2, 9, 120, 160])
        
        x = self.decoder2(x, ff2)
        x = F.upsample(x, scale_factor=2, mode='nearest') # stage2
        semantic2 = self.classifier2(x) # torch.Size([2, 9, 240, 320])
        
        x = self.decoder3(x, ff3)
        x = F.upsample(x, scale_factor=2, mode='nearest') # stage3
        semantic = self.classifier3(x) # torch.Size([2, 9, 480, 640])
        

        return semantic, semantic2, sal, edge


def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,4,480,640).astype(np.float32))
    model = DGRFNet(n_class=9)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,9,480,640), 'output shape (2,9,480,640) is expected!'
    print('test ok!')
    torch.save(model, "cc.pth")


if __name__ == '__main__':
    # unit_test()
    device = 'cuda'
    num_minibatch = 1
    input1 = torch.randn(1, 3, 480, 640).cuda(0)
    input2 = torch.randn(1, 1, 480, 640).cuda(0)
    model = DGRFNet(9).cuda(0)
    macs, params = profile(model, inputs=(input1, input2), 
                    custom_ops={model: model.forward})
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)