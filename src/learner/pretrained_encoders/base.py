import torch
from torch.nn import Sequential, AdaptiveAvgPool2d, Identity, Module
from typing import Iterable

from torch.nn.modules.flatten import Flatten

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
