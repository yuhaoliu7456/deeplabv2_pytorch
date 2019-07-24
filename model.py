import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import _layer1, _ResLayer, _Conv_Bn_Relu


class _ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, atrous_rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(atrous_rates):
            self.add_module("c%d" %i, nn.Conv2d(in_channel, out_channel, 3, 1, padding=rate, dilation=rate, bias=True))
        
        for m in self.children():
            nn.init.normal_(m.weight.data, mean=0, std=0.01)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        # print(list(self.children())[0](x) == list(self.children())[1](x))  #判断x从不同扩张率的卷积出来的结果是否一样
        sum_output = sum([stage(x) for stage in self.children()])
        return sum_output


class Deeplabv2(nn.Sequential):
    def __init__(self, num_classes, num_blocks, atrous_rates):
        super(Deeplabv2, self).__init__()
        channels = [64 * 2**i for i in range(6)]
        
        self.add_module("layer1", _layer1(channels[0]))
        self.add_module("layer2", _ResLayer(num_blocks[0], channels[0], channels[2], 1, 1))
        self.add_module("layer3", _ResLayer(num_blocks[1], channels[2], channels[3], 2, 1))   # the reslayer'input dim must be carefully, i made a mistake here
        self.add_module("layer4", _ResLayer(num_blocks[2], channels[3], channels[4], 1, 2))
        self.add_module("layer5", _ResLayer(num_blocks[3], channels[4], channels[5], 1, 4))
        self.add_module("ASPP", _ASPP(channels[5], num_classes, atrous_rates))
    
    def freeze_bn(self):
        # 把在coco上pretrained的model的conv中的BN给设成eval()模式以不让它训练
        for i in self.modules():
            if isinstance(i, _Conv_Bn_Relu.BATCH_NORM):
                i.eval() 
    
class Model(nn.Module):
    def __init__(self, base, scales):
        super(Model, self).__init__()
        self.scales = scales
        self.base = base
    
    def forward(self, x):
        logits = self.base(x)
        #1. obtain the orginal output's height and width
        _, _, height, width = logits.shape

        #2. obtain the scaled input x_
        logits_pyramid = []
        for scale in self.scales:
            x_ = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(x_))

        #3. cal all logits
        logits_all = [logits] + [F.interpolate(l, size=(height, width), mode="bilinear", align_corners=False) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max
         


if __name__ == "__main__":
    rate = list((6, 12, 18, 24))
    model = Model(base=Deeplabv2(num_classes=21, num_blocks=[3, 4, 23, 3], atrous_rates=rate), scales=(0.5, 0.75))
    model.eval()
    x = torch.randn(1, 3, 30, 30)
    result = model(x)
    print(result)

