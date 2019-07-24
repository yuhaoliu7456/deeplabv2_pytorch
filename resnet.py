from __future__ import absolute_import, print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Sequential):   # Sequential是一个特殊的module，它包含几个子Module,所以可以直接用self.add_module()
    def __init__(self, num_class, n_blocks):
        super(Resnet, self).__init__()
        channels = [64 * 2 **p for p in range(6)]
        self.add_module("layer1", _layer1(channels[0]))    
        self.add_module("layer2", _ResLayer(n_blocks[0], channels[0], channels[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], channels[2], channels[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], channels[3], channels[4], 2, 1))
        self.add_module("layer5", _ResLayer(n_blocks[3], channels[4], channels[5], 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(output_size=1))   # 只需要传入输出大小，不需要管stride，在SPP中就是这
        self.add_module("flatten", _Flatten())
        self.add_module("fc", nn.Linear(channels[5], num_class))
        # method2 to initial model's param
        # self._init_weight_method2()

    def _init_weight_method2(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)   # I'm not sure this two lines code
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

class _layer1(nn.Sequential):
    def __init__(self, out_channel):
        super(_layer1, self).__init__()
        self.add_module("conv1", _Conv_Bn_Relu(in_channel=3, out_channel=out_channel, kernel_size=7, stride=2, padding=3, dilation=1, relu=True))
        self.add_module("pool1", nn.MaxPool2d(3, 2, 1, ceil_mode=True))

class _ResLayer(nn.Sequential):
    def __init__(self, num_layer, in_channel, out_channel, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(num_layer)]
        else:
            assert num_layer == len(multi_grids)
        # downsampling is only in the first block
        for i in range(num_layer):
            self.add_module("block{}".format(i + 1),
                            _Bottleneck(in_channel=in_channel if i==0 else out_channel,
                                        out_channel=out_channel,
                                        stride=(stride if i ==0 else 1),
                                        dilation=dilation * multi_grids[i],
                                        downsample = (True if i == 0 else False)))

class _Bottleneck(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.downsample = downsample
        middle_channel = out_channel // 4
        # 从layer3--layer5之间，每个layer的第一个block的第一个conv操作会reduce resolution
        self.reduce = _Conv_Bn_Relu(in_channel=in_channel, out_channel=middle_channel, kernel_size=1, stride=stride, padding=0, dilation=1, relu=True)
        self.conv3x3 = _Conv_Bn_Relu(in_channel=middle_channel, out_channel=middle_channel, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)
        self.increase = _Conv_Bn_Relu(in_channel=middle_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=0, dilation=1, relu=False)
        self.shortcut = (
            _Conv_Bn_Relu(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=stride, padding=0, dilation=1, relu=False)
            if downsample 
            else lambda x:x)  #只在每个layer的第一个block做downsampling，这里做一个卷积的意思是为了输入这个layer的input和输出这个layer第一个block的维度相同；
                              # 之后输入这个layer里的其它block的input和output都是相同维度，即不需要downsampling
    
    def forward(self, x):
        return F.relu(self.increase(self.conv3x3(self.reduce(x))) + self.shortcut(x))

class _Conv_Bn_Relu(nn.Sequential):
    BATCH_NORM = nn.BatchNorm2d 
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation, relu=True):
        super(_Conv_Bn_Relu, self).__init__()
        self.add_module("conv", nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module("relu", nn.ReLU())

class _Flatten(nn.Module):
    def forward(self, x):
        # x.size() [1, 2048, 1, 1]
        
        return x.view(x.size(0), -1)



if __name__ == "__main__":
    model = Resnet101(num_class=200, n_blocks=[3, 4, 6, 3])
    print(model.layer1)
    input_ = torch.randn(1, 3, 224, 224)
    output_ = model(input_)
    # print(output_)
    # for i in model.named_parameters():
    #     print(type(i.data), i.size())
    # print(model.state_dict().keys())