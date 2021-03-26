import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .modules import *


class DualPathwayNetwork(nn.Module):
    def __init__(self, n_classes, n_channels=3, layers=[2, 2, 2, 2], planes=32):
        super(DualPathwayNetwork, self).__init__()

        self.steam =  nn.Sequential(nn.Conv2d(n_channels, planes, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(planes),
                                    Swish(),
                                    nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(planes),
                                    Swish(),
                                    )
        block = InvertedResidualBlock    
        highres_planes = planes * 2

        self.layer1 = make_layer(block, planes, planes, layers[0])
        self.layer2 = make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 = make_layer(block, planes * 8, planes * 8, 1, stride=2)

        self.hr_layer3 = make_layer(block, planes * 2, highres_planes, 2)
        self.hr_layer4 = make_layer(block, highres_planes, highres_planes, 2)
        self.hr_layer5 = make_layer(block, highres_planes, highres_planes, 1)
        
        self.bfm_1 = BilateralFusionModule(planes * 4, highres_planes, stage=1)
        self.bfm_2 = BilateralFusionModule(planes * 8, highres_planes, stage=2)

        self.context_aggregation_block = PSPModule(planes * 8, planes * 4, activation=Swish)
        self.segmentation_head = SegmentationHead(planes * 4, n_classes, scale=8) 

        self.aux_head_1 = SegmentationHead(highres_planes, n_classes, scale=8) 
        self.aux_head_2 = SegmentationHead(highres_planes, n_classes, scale=8) 

        self.projection = nn.Conv2d(planes * 4 + highres_planes, planes * 4, 1, bias=False)

        init_weight(self)

    def forward(self, x):
        x = self.steam(x)
 
        x = self.layer1(x)
        x = self.layer2(x)

        hx = self.hr_layer3(x)
        x = self.layer3(x)

        x, hx = self.bfm_1(x, hx)
        
        aux_1 = hx

        x = self.layer4(x)
        hx = self.hr_layer4(hx)
        
        x, hx = self.bfm_2(x, hx)

        aux_2 = hx

        x = self.layer5(x)
        hx = self.hr_layer5(hx)

        x = self.context_aggregation_block(x)

        _, _, H, W = hx.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        feature_sum = self.projection(torch.cat([x, hx], dim=1))

        head = self.segmentation_head(feature_sum)

        if self.training:
            aux_head_1 = self.aux_head_1(aux_1)
            aux_head_2 = self.aux_head_2(aux_2)

            return head, aux_head_1, aux_head_2

        return head


def get_network(n_classes):
    net = DualPathwayNetwork(n_classes=n_classes)
    
    if torch.cuda.is_available():
        net = net.cuda()

    net.train()

    return net


def main():
    net = get_network(n_classes=21)
    net.train()

    in_ten = torch.randn(2, 3, 256, 256)

    if torch.cuda.is_available():
        net = net.cuda()
        in_ten = in_ten.cuda()
    
    out = net(in_ten)
    print(out.shape)


if __name__ == "__main__":
    main()