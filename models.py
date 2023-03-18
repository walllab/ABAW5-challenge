import timm
import torch.nn as nn
import torch.nn.functional as F

class headClass(nn.Module):
    def __init__(self, layer_widths, output_dim = 7, dropout=False):
        super().__init__()
        
        self.layer_widths = layer_widths
        layers = []
        if len(layer_widths)>1:
            for i, w in enumerate(layer_widths[:-1]):
                layers.append(nn.Linear(layer_widths[i], layer_widths[i+1], bias=True))
                layers.append(nn.ReLU(inplace=True))
                
        output_layer = nn.Linear(layer_widths[-1], output_dim, bias=False) 
        output_layer = nn.utils.weight_norm(output_layer)
        layers.append(output_layer)       
        self.layers = nn.ModuleList(layers)
        
        if dropout:
            self.dropout = nn.Dropout()
        else:
            self.dropout = None
        
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.normalize(x, dim=-1)
        x = self.layers[-1](x)
        return x
    
class FEClassifier(nn.Module):
    def __init__(self, backbone, head_layers):
        super().__init__()
        
        if backbone=='resnet18':
            self._model = timm.create_model('resnet18d', pretrained=True)
            self._model.fc = headClass([512, *head_layers])

        elif backbone=='resnet50':
            self._model = timm.create_model('resnet50d', pretrained=True)
            self._model.fc = headClass([2048, *head_layers])

        elif backbone == 'seresnext101':
            self._model = timm.create_model('seresnext101_32x8d', pretrained=True)
            self._model.fc = headClass([2048, *head_layers])

    def forward(self, x):
        return self._model(x)
    
    def param_count(self):
        ct = 0
        for p in self._model.parameters():
            ct += len(p.flatten())
        print(f'Number of parameters: {ct}')