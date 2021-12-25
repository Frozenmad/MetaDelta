import timm
import torch.nn as nn
from timm.models.resnet import ResNet
from .base import SimpleSequentialModel

class ResNetWrap(SimpleSequentialModel):
    def __init__(self, model: ResNet):
        layer0 = nn.Sequential(model.conv1, model.bn1, model.act1, model.maxpool)
        super().__init__([layer0, model.layer1, model.layer2, model.layer3, model.layer4])

def rn_timm_mix(pretrained=True, name='ecaresnet50t', momentum=0.1):
    model = timm.create_model(name, pretrained=pretrained)
    model = ResNetWrap(model)
    print('model: rn_timm_mix, name:', name, 'layer num:', model.num_layers, 'momentum:', momentum)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum
    return model
