# Laywe-wise relu pruning

import torch
import torch.nn as nn

alpha = 1
rho = 1

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    # "resnet50",
    # "resnet101",
    # "resnet152",
    # "resnext50_32x4d",
    # "resnext101_32x8d",
    # "wide_resnet50_2",
    # "wide_resnet101_2",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride,
            downsample=None,
            groups=-1,
            groupsize=-1,
            residual=True,
            base_width=64,
            dilation=1,
            norm_layer=None,
            scale=16,
            enable_relu=True,
            thinned=False
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.residual = residual
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.enable_relu = enable_relu
        self.thinned = thinned
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if self.enable_relu:
            self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = None
        if self.residual:
            self.downsample = downsample
        self.stride = stride

        if self.enable_relu and (not self.thinned):
            self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.enable_relu:
            out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual and self.downsample is not None:
            identity = self.downsample(x)

        if self.residual:
            out += identity

        if self.enable_relu and (not self.thinned):
            out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes,
            groups,
            groupsize,
            residual,
            zero_init_residual=False,
            width_per_group=64,
            replace_stride_with_dilation=None,
            scale=16,
            norm_layer=None,
            **kwargs,
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        culled_val = kwargs['Culled']
        print(f'Culled Val: {culled_val}')
        if culled_val == 1:
            self.Culled = [False, True, True, True]
        elif culled_val == 14:
            self.Culled = [False, True, True, False]
        elif culled_val == 12:
            self.Culled = [False, False, True, True]
        elif culled_val == 123:
            self.Culled = [False, False, False, True]
        elif culled_val == 1234:
            self.Culled = [False, False, False, False]
        elif culled_val == 23:
            self.Culled = [True, False, False, True]
        elif culled_val == 124:
            self.Culled = [False, False, True, False]
        elif culled_val == 13:
            self.Culled = [False, True, False, True]
        elif culled_val == 134:
            self.Culled = [False, True, False, False]
        elif culled_val == 234:
            self.Culled = [True, False, False, False]
        elif culled_val == 0:
            self.Culled = [True, True, True, True]
        self.Thinned = kwargs['Thinned']
        self.alpha = kwargs['alpha']
        self.rho = kwargs['rho']


        self.groups = groups
        self.scale = scale
        self.groupsize = groupsize
        self.residual = residual

        self.inplanes = int(self.alpha * 64)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.ChMulFact = [2, 2, 2]
        # self.Culled = [False, True, True, True]

        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=int(1 // self.rho), padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)

        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.inplanes, layers[0], enable_relu=self.Culled[0], thinned=self.Thinned)
        self.layer2 = self._make_layer(block, self.inplanes * self.ChMulFact[0], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], enable_relu=self.Culled[1], thinned=self.Thinned)

        self.layer3 = self._make_layer(block, self.inplanes * self.ChMulFact[1], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], enable_relu=self.Culled[2], thinned=self.Thinned)

        self.layer4 = self._make_layer(block, self.inplanes * self.ChMulFact[2], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], enable_relu=self.Culled[3], thinned=self.Thinned)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                # 	nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                # 	nn.init.constant_(m.bn2.weight, 0)
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, enable_relu=True, thinned=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.groupsize,
                self.residual,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.scale,
                enable_relu,
                thinned
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    groups=self.groups,
                    groupsize=self.groupsize,
                    residual=self.residual,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    scale=self.scale,
                    enable_relu=enable_relu,
                    thinned=thinned
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
        arch,
        block,
        layers,
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
):
    model = ResNet(block, layers, num_classes, groups, groupsize, residual, **kwargs)
    return model


def resnet18(
        num_classes, groups, groupsize, residual, pretrained=False, progress=True, **kwargs
):
    # if groups == 0:
    # 	block = BasicBlock
    # else:
    # 	block = BasicBlockDWS
    block = BasicBlock
    return _resnet(
        "resnet18",
        block,
        [2, 2, 2, 2],
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
    )


def resnet34(
        num_classes, groups, groupsize, residual, pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
    )

# def resnet50(
# 	num_classes, groups, groupsize, residual, pretrained=False, progress=True, **kwargs
# ):
# 	return _resnet(
# 		"resnet50",
# 		Bottleneck,
# 		[3, 4, 6, 3],
# 		num_classes,
# 		groups,
# 		groupsize,
# 		residual,
# 		pretrained,
# 		progress,
# 		**kwargs
# 	)
#
#
# def resnet101(pretrained=False, progress=True, **kwargs):
# 	return _resnet(
# 		"resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
# 	)
#
#
# def resnet152(pretrained=False, progress=True, **kwargs):
# 	return _resnet(
# 		"resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
# 	)
#
#
# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
# 	kwargs["groups"] = 32
# 	kwargs["width_per_group"] = 4
# 	return _resnet(
# 		"resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
# 	)
#
#
# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
# 	kwargs["groups"] = 32
# 	kwargs["width_per_group"] = 8
# 	return _resnet(
# 		"resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
# 	)
#
#
# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
# 	kwargs["width_per_group"] = 64 * 2
# 	return _resnet(
# 		"wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
# 	)
#
#
# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
# 	kwargs["width_per_group"] = 64 * 2
# 	return _resnet(
# 		"wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
# 	)
