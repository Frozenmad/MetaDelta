import timm
from torch.nn import BatchNorm2d
from timm.models.efficientnet import EfficientNet
from .base import SequentialModel

class ENetWrapper(SequentialModel):
    def __init__(self, model: EfficientNet):
        super().__init__(2 + len(model.blocks))
        self.model = model
        print('layer num', self.num_layers)
    
    def layer_forward(self, x, layer_id=0):
        if layer_id == 0:
            x = self.model.conv_stem(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
        elif layer_id == self.num_layers - 1:
            x = self.model.conv_head(x)
            x = self.model.bn2(x)
            x = self.model.act2(x)
        else:
            x = self.model.blocks[layer_id - 1](x)
        return x

    def finalize(self, x, layer_id=0):
        return self.model.global_pool(x)
    
    def get_parameters(self, layer=0):
        parameters = []
        if layer == 0:
            parameters.extend(list(self.model.conv_stem.parameters()))
            parameters.extend(list(self.model.bn1.parameters()))
            parameters.extend(list(self.model.act1.parameters()))
        elif layer == self.num_layers - 1:
            parameters.extend(list(self.model.conv_head.parameters()))
            parameters.extend(list(self.model.bn2.parameters()))
            parameters.extend(list(self.model.act2.parameters()))
        elif layer == self.num_layers:
            return []
        else:
            parameters.extend(list(self.model.blocks[layer - 1].parameters()))
        return parameters

    def set_layer(self, layer, learnable):
        super().set_layer(layer, learnable)
        # set the corresponding layer to eval mode if not learnable
        if not learnable:
            if layer == 0:
                self.model.conv_stem.eval()
                self.model.bn1.eval()
                self.model.act1.eval()
            elif layer == self.num_layers - 1:
                self.model.conv_head.eval()
                self.model.bn2.eval()
                self.model.act2.eval()
            elif layer == self.num_layers:
                pass
            else:
                self.model.blocks[layer - 1].eval()

    def set_mode(self, train=True):
        if train:
            for layer in range(self.num_layers + 1):
                if self.layer_learnable[layer]:
                    if layer == 0:
                        self.model.conv_stem.train()
                        self.model.bn1.train()
                        self.model.act1.train()
                    elif layer == self.num_layers - 1:
                        self.model.conv_head.train()
                        self.model.bn2.train()
                        self.model.act2.train()
                    elif layer == self.num_layers:
                        pass
                    else:
                        self.model.blocks[layer - 1].train()
        else:
            self.eval()

def enet_mixup(pretrained=True, name='tf_efficientnet_b0_ns', momentum=0.1, *args, **kwargs):
    model = timm.create_model(name, pretrained=pretrained)
    del model.classifier
    wrapper = ENetWrapper(model)
    for module in wrapper.modules():
        if isinstance(module, BatchNorm2d):
            module.momentum = momentum
    print('enet model of name', name, 'layer size', wrapper.num_layers, 'with momentum', momentum)
    return wrapper
