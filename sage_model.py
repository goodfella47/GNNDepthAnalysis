import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, PairNorm


class SAGE(torch.nn.Module):
    
    def __init__(self, in_channels, 
                 hidden_channels, out_channels,
                 n_layers=2):
        super(SAGE, self).__init__()
        
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList()

        if n_layers == 1:
            self.layers.append(SAGEConv(in_channels, out_channels, normalize=False))
        elif n_layers == 2:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        else:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(PairNorm())

            for _ in range(n_layers - 2):
                self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
                self.layers_bn.append(PairNorm())

            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, depth):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers

        for i, layer in enumerate(looper[-depth:]):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
                # a = 2
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)

        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)

        return x
