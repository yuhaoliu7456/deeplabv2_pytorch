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
        sum_output = sum([stage(x) for stage in self.children()])
        return sum_output


class Deeplabv2(nn.Sequential):
    def __init__(self, num_classes, num_blocks, atrous_rates):
        super(Deeplabv2, self).__init__()
        channels = [64 * 2**i for i in range(6)]
        
        self.add_module("layer1", _layer1(channels[0]))
        self.add_module("layer2", _ResLayer(num_blocks[0], channels[0], channels[2], 1, 1))
        self.add_module("layer3", _ResLayer(num_blocks[1], channels[2], channels[3], 2, 1))   
        self.add_module("layer4", _ResLayer(num_blocks[2], channels[3], channels[4], 1, 2))
        self.add_module("layer5", _ResLayer(num_blocks[3], channels[4], channels[5], 1, 4))
        self.add_module("ASPP", _ASPP(channels[5], num_classes, atrous_rates))
    
    def freeze_bn(self):
        # Set the BN in the conv of the pretrained model on coco to eval() mode to prevent it from training.
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
 
def _init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)



