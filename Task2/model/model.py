import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def init_weight(self, gain=0.02):
    for ly in self.children():
        if isinstance(ly, nn.Conv2d):
            nn.init.normal_(ly.weight.data, 0.0, gain)

            if not ly.bias is None: nn.init.constant_(ly.bias, 0)

        elif isinstance(ly, nn.BatchNorm2d):
            nn.init.normal_(ly.weight.data, 1.0, gain)
            nn.init.constant_(ly.bias.data, 0.0)


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()

        self.up_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),  # anti-aliasing filter
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        init_weight(self)

    def forward(self, x):
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/5
        x = F.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)
        return self.up_conv(x)


class RCLBlockB(nn.Module):
    """
    https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
    """
    def __init__(self, ch_in, ch_out, t):
        super(RCLBlockB, self).__init__()
        self.t = t

        self.ch_in = ch_in
        self.ch_out = ch_out

        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        self.conv_2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn_ac_list = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(ch_out), nn.ReLU()) for _ in range(0, self.t)])

        init_weight(self)

    def forward(self, x):
        stack_t = self.conv[0](x)
        stack_tn = self.conv[1:](stack_t)  # tn = t0

        # TODO: need to make it into a loop, to support arbitrary t.
        stack_tn = self.bn_ac_list[0](stack_t + self.conv_2(stack_tn))  # tn = t1
        stack_tn = self.bn_ac_list[1](stack_t + self.conv_2(stack_tn))  # tn = t2
        stack_tn = self.bn_ac_list[2](stack_t + self.conv_2(stack_tn))  # tn = t3

        return stack_tn


class RRCNNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNNBlock, self).__init__()

        self.conv_1x1 = nn.Conv2d(ch_in, ch_out, 1)  # dim reduction

        self.rcnn = nn.Sequential(
            RCLBlockB(ch_in, ch_out, t=t),
            RCLBlockB(ch_out, ch_out, t=t)
        )

        init_weight(self)

    def forward(self, x):
        y = self.conv_1x1(x)

        return self.rcnn(y) + y


class R2UNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, t=3, ratio=1/4):
        """
        3 -> 64 -> 128 -> 256 -> 512 -> 256  ->  128 -> 64 -> 19         ---- t=3, ratio = 1,   17.5M
        3 -> 16 -> 32 -> 64 -> 128 -> 64 –> 32 -> 16 -> 19               ---- t=3, ratio = 0.25, 1.1M
        """
        super(R2UNet, self).__init__()

        self.encoding_filters = [int(64 * ratio), int(128 * ratio), int(256 * ratio), int(512 * ratio)]

        self.stem = RRCNNBlock(ch_in=n_channels, ch_out=self.encoding_filters[0], t=t)

        self.en_stages = nn.ModuleList([nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                                      RRCNNBlock(ch_in=self.encoding_filters[i],
                                                                 ch_out=self.encoding_filters[i + 1],
                                                                 t=t)
                                                     )
                                       for i in range(len(self.encoding_filters) - 1)])

        self.decoding_filters = [int(512 * ratio), int(256 * ratio), int(128 * ratio), int(64 * ratio)]
        self.up_convs = nn.ModuleList([UpConv(ch_in=self.decoding_filters[i],
                                              ch_out=self.decoding_filters[i + 1],
                                              ) for i in range(len(self.decoding_filters) - 1)])

        self.de_stages = nn.ModuleList([RRCNNBlock(ch_in=self.decoding_filters[i],
                                                   ch_out=self.decoding_filters[i + 1],
                                                   t=t) for i in range(len(self.decoding_filters) - 1)])
        init_weight(self)

        self.segmentation_head = nn.Conv2d(self.decoding_filters[-1], n_classes, 1)
        # nn.ReLU(inplace=True)  # not advisable since in pytorch cross-entropy uses log sigmoid.

    def forward(self, x):
        e1 = self.stem(x)

        e2 = self.en_stages[0](e1)
        e3 = self.en_stages[1](e2)
        e4 = self.en_stages[2](e3)

        d4 = torch.cat((e3, self.up_convs[0](e4)), dim=1)
        d4 = self.de_stages[0](d4)

        d3 = torch.cat((e2, self.up_convs[1](d4)), dim=1)
        d3 = self.de_stages[1](d3)

        d2 = torch.cat((e1, self.up_convs[2](d3)), dim=1)
        d2 = self.de_stages[2](d2)

        head = self.segmentation_head(d2)

        return head


def get_network(n_classes):
    net = R2UNet(n_classes=n_classes)
    
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
