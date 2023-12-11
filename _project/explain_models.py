import torch
import torch_geometric
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GPSConv, Sequential, GINEConv
from torch.nn import Embedding

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes, h_dim=32):
        super().__init__()
        self.conv1 = GCNConv(n_features, h_dim)
        self.conv2 = GCNConv(h_dim, h_dim)
        self.conv3 = GCNConv(h_dim, h_dim)
        self.lin = torch.nn.Linear(h_dim, n_classes)

    def forward(self, x, edge_index, batch_mapping=None, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)


        #converts batch (which is apparently a merged graph of all the indivudal graphs in batch) back into regular format with batch_size x outs
        if batch_mapping is None:
            x = global_mean_pool(x, batch=None)
        else:
            x = global_mean_pool(x, batch_mapping)

        x = self.lin(x)
        return torch.nn.functional.log_softmax(x, dim=1)

class GPS_Model(torch.nn.Module):
    def __init__(self, n_features, n_classes, embd_dim=16, n_embds=32, h_dim=32, n_layers=1, n_heads=4):
        super().__init__()
        assert n_layers >= 1, "Error: Must have at least one layer!"

        #self.node_embedd = Embedding(n_embds, embd_dim)
        self.node_feature_stretch = torch.nn.Linear(n_features, h_dim)
        self.lin = torch.nn.Linear(h_dim, n_classes)

        #self.first_conv = GPSConv(n_features, conv=GCN(), heads=8)
        self.list_convs = ModuleList()
        for i in range(n_layers):
            #attn_type = 'performer'
            attn_type = 'multihead'
            #layer_conv = GPSConv(channels=n_features, conv=GINEConv(gine_contents), heads=n_heads, attn_type=attn_type)
            #
            # if i == 0:
            #     layer_conv = GPSConv(channels=n_features, conv=GCNConv(in_channels=n_features, out_channels=n_features), heads=1, attn_type=attn_type)
            # else:
            #     layer_conv = GPSConv(channels=h_dim, conv=GCNConv(in_channels=h_dim, out_channels=h_dim), heads=n_heads, attn_type=attn_type)

            layer_conv = GPSConv(channels=h_dim, conv=GCNConv(in_channels=h_dim, out_channels=h_dim), heads=1,
                                 attn_type=attn_type)
            self.list_convs.append(layer_conv)


    def forward(self, x, edge_index, batch_mapping=None, edge_attr=None):

        #x = torch.nn.functional.dropout(x, training=self.training)
        if not batch_mapping is None:
            x = torch.stack(torch_geometric.utils.unbatch(x, batch_mapping), dim=0)
        x = self.node_feature_stretch(x)
        x = x.reshape(-1, x.shape[-1])

        for conv in self.list_convs:
            #x = self.node_embedd(x)
            #x = self.node_feature_stretch(x)

            if not batch_mapping is None:
                if not edge_attr is None:
                    x = conv(x, edge_index, batch_mapping, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index, batch_mapping)
            else:
                x = conv(x, edge_index)

        if not batch_mapping is None:
            x = global_mean_pool(x, batch_mapping)
        x = self.lin(x)
        return torch.nn.functional.log_softmax(x, dim=1)