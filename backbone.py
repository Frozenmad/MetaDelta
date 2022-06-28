import torch
import torch.nn as nn
from torch.nn import Sequential, AdaptiveAvgPool2d, Identity, Module
from typing import Iterable
from torch.nn.modules.flatten import Flatten
import timm
from timm.models.resnet import ResNet

def normalize(x):
    return x / (1e-6 + x.pow(2).sum(dim=-1, keepdim=True).sqrt())

class MLP(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.core = nn.Sequential([
            nn.Linear(indim, indim // 2), nn.ReLU(),
            nn.Linear(indim // 2, indim // 2), nn.ReLU(),
            nn.Linear(indim // 2, outdim)
        ])

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.core(x)


class SequentialModel(Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layer_learnable = {}
        for i in range(self.num_layers + 1):
            self.layer_learnable[i] = True

    def layer_forward(self, x, layer_id=0):
        raise NotImplementedError

    def finalize(self, x, layer_id):
        raise NotImplementedError
    
    def get_parameters(self, layer) -> Iterable[torch.nn.Parameter]:
        raise NotImplementedError

    def forward(self, x):
        for i in range(self.num_layers):
            if self.layer_learnable[i]:
                x = self.layer_forward(x, i)
            else:
                with torch.no_grad():
                    x = self.layer_forward(x, i)
        if self.layer_learnable[self.num_layers]:
            x = self.finalize(x, self.num_layers)
        else:
            with torch.no_grad():
                x = self.finalize(x, self.num_layers)
        return x

    def set_layer(self, layer, learnable=True):
        self.layer_learnable[layer] = learnable

    def set_mode(self, train):
        raise NotImplementedError

class SimpleSequentialModel(SequentialModel):
    def __init__(self, models: list, last_layer: Module = Identity(), final_layer : Module =Sequential(AdaptiveAvgPool2d((1,1)), Flatten())):
        super().__init__(len(models))
        self.core = Sequential(*models)
        self.final = final_layer
        self.last = last_layer
    
    def layer_forward(self, x, layer_id=0):
        return self.core[layer_id](x)
    
    def finalize(self, x, layer_id):
        if layer_id == self.num_layers: x = self.last(x)
        return self.final(x)

    def get_parameters(self, layer):
        if layer == self.num_layers: return list(self.last.parameters())
        return list(self.core[layer].parameters())

    def set_mode(self, train):
        if not train: self.eval()
        else:
            for i in range(self.num_layers):
                if self.layer_learnable[i]: self.core[i].train()
                else: self.core[i].eval()
            if self.layer_learnable[self.num_layers]: self.last.train()
            else: self.last.eval()
            self.final.train()


class Wrapper(Module):
    def __init__(self, model: SequentialModel):
        super().__init__()
        self.model = model
        self.num_layers = self.model.num_layers
        self.set = False
    
    def set_get_trainable_parameters(self, parameters=[]):
        params = []
        for i in range(self.num_layers + 1):
            param = self.model.get_parameters(layer=i)
            if i in parameters:
                params.extend(param)
            elif not self.set:
                for p in param:
                    p.requires_grad = False
        self.set = True
        return params
    
    def set_learnable_layers(self, layers):
        for i in range(self.num_layers + 1):
            self.model.set_layer(i, i in layers)
    
    def set_mode(self, train):
        self.model.set_mode(train)

    def forward(self, x):
        for layer in range(self.num_layers):
            x = self.model.layer_forward(x, layer)
        return self.model.finalize(x, layer + 1)

class ResNetWrap(SimpleSequentialModel):
    def __init__(self, model: ResNet):
        layer0 = nn.Sequential(model.conv1, model.bn1, model.act1, model.maxpool)
        super().__init__([layer0, model.layer1, model.layer2, model.layer3, model.layer4])

def rn_timm_mix(pretrained=True, name='swsl_resnet50', momentum=0.1):
    model = timm.create_model(name, pretrained=pretrained)
    model = ResNetWrap(model)
    print('model: rn_timm_mix, name:', name, 'layer num:', model.num_layers, 'momentum:', momentum)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = momentum
    return model
