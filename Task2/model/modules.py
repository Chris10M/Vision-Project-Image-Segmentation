import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(v: int, divisor: int = 8, min_value: int = None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # ensure round down does not go down by more than 10%.
        new_v += divisor
    return new_v


def init_weight(self, gain=0.02):
    for ly in self.children():
        if isinstance(ly, nn.Conv2d):
            nn.init.normal_(ly.weight.data, 0.0, gain)

            if not ly.bias is None: nn.init.constant_(ly.bias, 0)

        elif isinstance(ly, nn.BatchNorm2d):
            nn.init.normal_(ly.weight.data, 1.0, gain)
            nn.init.constant_(ly.bias.data, 0.0)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SqueezeExcite(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcite, self).__init__()

        self.conv_reduce = nn.Conv2d(channel, channel // reduction, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(channel // reduction, channel, 1, bias=True)

        self.gate_fn = nn.Sigmoid()

        init_weight(self)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)

        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, exp_ratio=4, residual=True):
        super(InvertedResidualBlock, self).__init__()

        self.residual = residual

        if residual:
            if stride == 1 and in_chs == out_chs:
                self.down_sample = nn.Identity()
            else:
                self.down_sample = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False)

        mid_chs: int = make_divisible(in_chs * exp_ratio)

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_chs, mid_chs, 1)
        self.bn1 = nn.BatchNorm2d(mid_chs)
        self.act1 = Swish()

        # Depth-wise convolution
        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, 3, padding=1, groups=mid_chs, stride=stride)

        self.bn2 = nn.BatchNorm2d(mid_chs)
        self.act2 = Swish()

        self.se = SqueezeExcite(mid_chs)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv2d(mid_chs, out_chs, 1)
        self.bn3 = nn.BatchNorm2d(out_chs)

        init_weight(self)

    def forward(self, x):
        residual = self.down_sample(x)

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.residual:
            x += residual

        return x


def make_layer(block, inplanes, planes, blocks, stride=1):
    layers = []

    layers.append(block(inplanes, planes, stride))

    for i in range(1, blocks):
        layers.append(block(planes, planes, stride=1))
            
    return nn.Sequential(*layers)


class BilateralFusionModule(nn.Module):
    def __init__(self, low_plane, high_plane, stage=1):
        super(BilateralFusionModule, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.low_to_high = nn.Sequential(nn.Conv2d(low_plane, high_plane, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(high_plane),
                                         )

        self.high_to_low = nn.Sequential(nn.Conv2d(high_plane, low_plane,
                                                   kernel_size=3 + (stage - 1) * 2,
                                                   stride=2*stage, padding=1 + (stage - 1),
                                                   bias=False),
                                         nn.BatchNorm2d(low_plane)
                                         )

        self.swish = Swish()        

        init_weight(self)

    def forward(self, low_res, high_res):
        _, _, H, W = high_res.shape

        hif = F.interpolate(self.low_to_high(low_res), size=(H, W), align_corners = True, mode='bilinear')
        lif = self.high_to_low(high_res)

        lf = self.swish(low_res + lif)
        hf = self.swish(high_res + hif)

        return lf, hf


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, n_classes, expansion=1, scale=None): 
        super(SegmentationHead, self).__init__()
        
        self.fc = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Swish(),
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=3, padding=1, bias=False),
            
            nn.BatchNorm2d(in_channels * expansion),
            Swish(),
            nn.Conv2d(in_channels * expansion, n_classes, kernel_size=1),
        )
        self.scale = scale

        init_weight(self)

    def forward(self, x):
        x = self.fc(x)

        if self.scale is not None:
            _, _, H, W = x.shape
            x = F.interpolate(x, size=(H * self.scale, W * self.scale), mode='bilinear', align_corners=True)

        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, n_classes, expansion=4):
        super(ClassificationHead, self).__init__()

        self.fc = nn.Sequential(
                                nn.BatchNorm2d(in_channels),
                                Swish(),
                                nn.Conv2d(in_channels, 32 * expansion, kernel_size=3, padding=1, bias=False),
            
                                nn.AdaptiveAvgPool2d((1, 1)),
                            )

        self.linear = nn.Linear(32 * expansion, n_classes)

    def forward(self, x):
        B, _, _, _ = x.shape 
        
        x = self.fc(x)
        return self.linear(x.view(B, -1))


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6), activation=nn.ReLU):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.activation = activation()

        init_weight(self)


    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)

        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages] + [feats]

        bottle = self.bottleneck(torch.cat(priors, 1))

        return self.activation(bottle)


# class _ASPPModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
#         super(_ASPPModule, self).__init__()
#         self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
#                                             stride=1, padding=padding, dilation=dilation, bias=False)
#         self.bn = BatchNorm(planes)
#         self.relu = nn.ReLU()

#         self._init_weight()

#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)

#         return self.relu(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class ASPP(nn.Module):
#     def __init__(self, backbone, output_stride, BatchNorm):
#         super(ASPP, self).__init__()
#         if backbone == 'drn':
#             inplanes = 512
#         elif backbone == 'mobilenet':
#             inplanes = 320
#         else:
#             inplanes = 2048
#         if output_stride == 16:
#             dilations = [1, 6, 12, 18]
#         elif output_stride == 8:
#             dilations = [1, 12, 24, 36]
#         else:
#             raise NotImplementedError

#         self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
#         self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
#         self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
#         self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
#                                              BatchNorm(256),
#                                              nn.ReLU())
#         self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
#         self.bn1 = BatchNorm(256)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()

#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         return self.dropout(x)



class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                                                    size=[height, width],
                                                    mode='bilinear') + x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                                                   size=[height, width],
                                                   mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class SelfAttentionModulle(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width ,height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )

        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x

        return out, attention


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, activation=nn.ReLU):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            activation(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            activation(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class AttentionPSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6), activation=nn.ReLU):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        
        in_chns = out_features

        self.query_conv = nn.Conv2d(in_channels=in_chns, out_channels=in_chns//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels=in_chns, out_channels=in_chns//8, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels=in_chns, out_channels=in_chns, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        self.bottleneck = GhostModule(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.activation = activation()
        
        init_weight(self)


    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = GhostModule(features, features, 3, dw_size=5, activation=Swish)
        
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)

        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages] + [feats]
        x = torch.cat(priors, 1)

        x = self.bottleneck(x)

        m_batchsize, C, width ,height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))

        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x

        return self.activation(out)

