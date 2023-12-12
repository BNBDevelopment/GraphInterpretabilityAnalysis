import torch
import torch_geometric
from torch.nn import ModuleList, ReLU, Linear, GRU
from torch.nn.functional import elu
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GPSConv, Sequential, GINEConv, TransformerConv, \
    NNConv, Set2Set
from torch.nn import Embedding

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes, h_dim=32):
        super().__init__(do_embedding=False)
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
        return torch.nn.functional.softmax(x, dim=1)

class GPS_Model(torch.nn.Module):
    def __init__(self, n_features, n_classes, embd_dim=16, n_embds=32, h_dim=32, n_layers=1, n_heads=4, do_embedding=False):
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
        return torch.nn.functional.softmax(x, dim=1)



class Attention_Model(torch.nn.Module):
    def __init__(self, n_features, n_classes, embd_dim=16, n_embds=32, h_dim=32, n_layers=1, n_heads=4, do_embedding=False):
        super().__init__()
        assert n_layers >= 1, "Error: Must have at least one layer!"

        self.node_feature_stretch = torch.nn.Linear(n_features, h_dim)
        self.lin = torch.nn.Linear(h_dim, n_classes)

        self.list_convs = ModuleList()
        for i in range(n_layers):
            #attn_type = 'performer'
            attn_type = 'multihead'
            layer_conv = GPSConv(channels=h_dim, conv=TransformerConv(in_channels=h_dim, out_channels=h_dim, heads=n_heads), heads=n_heads,
                                 attn_type=attn_type)
            self.list_convs.append(layer_conv)


    def forward(self, x, edge_index, batch_mapping=None, edge_attr=None):
        if batch_mapping is None:
            raise NotImplementedError("failure - need batch info")
        else:
            x = torch.stack(torch_geometric.utils.unbatch(x, batch_mapping), dim=0)
            x = self.node_feature_stretch(x)
            x = x.reshape(-1, x.shape[-1])
            for conv in self.list_convs:
                if not edge_attr is None:
                    x = conv(x, edge_index, batch_mapping, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index, batch_mapping)
            x = global_mean_pool(x, batch_mapping)
            x = self.lin(x)
            return torch.nn.functional.softmax(x, dim=1)


class AttentionConv_Model(torch.nn.Module):
    def __init__(self, n_features, n_classes, embd_dim=16, n_embds=32, h_dim=32, n_layers=1, n_heads=4, do_embedding=False):
        super().__init__()
        assert n_layers >= 1, "Error: Must have at least one layer!"

        self.node_feature_stretch = torch.nn.Linear(n_features, h_dim)
        self.lin = torch.nn.Linear(h_dim, n_classes)

        self.list_convs = ModuleList()
        for i in range(n_layers):
            #attn_type = 'performer'
            attn_type = 'multihead'
            layer_conv = TransformerConv(in_channels=h_dim, out_channels=h_dim//n_heads, heads=n_heads)
            self.list_convs.append(layer_conv)


    def forward(self, x, edge_index, batch_mapping=None, edge_attr=None):
        if batch_mapping is None:
            raise NotImplementedError("failure - need batch info")
        else:
            x = torch.stack(torch_geometric.utils.unbatch(x, batch_mapping), dim=0)
            x = self.node_feature_stretch(x)
            x = x.reshape(-1, x.shape[-1])
            for conv in self.list_convs:
                if not edge_attr is None:
                    x = conv(x, edge_index, batch_mapping, edge_attr=edge_attr)
                else:
                    x = conv(x=x, edge_index=edge_index)
            x = global_mean_pool(x, batch_mapping)
            x = self.lin(x)
            return torch.nn.functional.softmax(x, dim=1)


class TransformerConv2(torch.nn.Module):
    def __init__(self, n_features, n_classes, embd_dim=16, n_embds=38, h_dim=32, n_layers=1, n_heads=4, do_embedding=False, is_node_classification=False):
        super().__init__()

        self.n_classes = n_classes
        self.is_node_classification = is_node_classification


        self.conv1 = TransformerConv(in_channels=n_features, out_channels=h_dim // 2, heads=1)

        self.conv2 = TransformerConv(in_channels=(h_dim//2)*1, out_channels=h_dim//2, heads=1)
        self.lin1 = torch.nn.Linear((h_dim//2)*1, 64)
        self.lin2 = torch.nn.Linear(64, 128)
        self.lin3 = torch.nn.Linear(128, n_classes)



    def forward(self, x, edge_index, batch_mapping=None, edge_attr=None):
        x = self.conv1(x.to(torch.float32), edge_index)
        x = x.relu()
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = torch.nn.functional.dropout(x, training=self.training)


        #converts batch (which is apparently a merged graph of all the indivudal graphs in batch) back into regular format with batch_size x outs

        if self.is_node_classification:
            x = x
        else:
            if batch_mapping is None:
                x = global_mean_pool(x, batch=None)
            else:
                x = global_mean_pool(x, batch_mapping)

        x = self.lin1(x)
        x = elu(x)
        x = self.lin2(x)
        x = elu(x)
        x = self.lin3(x)
        x = elu(x)

        if self.n_classes == 1:
            return x
        elif self.n_classes == 10:
            return torch.nn.functional.log_softmax(x, dim=1)
        else:
            return torch.nn.functional.softmax(x, dim=1)


class CloneNet(torch.nn.Module):
    def __init__(self, n_features, n_classes, embd_dim=16, n_embds=38, h_dim=64, n_layers=1, n_heads=4, do_embedding=False, is_node_classification=False):
        super().__init__()
        self.lin0 = torch.nn.Linear(n_features, h_dim)

        nn = torch.nn.Sequential(Linear(h_dim, 128), ReLU(), Linear(128, h_dim * h_dim))
        self.conv = NNConv(h_dim, h_dim, nn, aggr='mean')
        self.gru = GRU(h_dim, h_dim)

        self.set2set = Set2Set(h_dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * h_dim, h_dim)
        self.lin2 = torch.nn.Linear(h_dim, 1)

    def forward(self, x, edge_index, batch, edge_attr):
        out = torch.nn.functional.relu(self.lin0(x))
        h = out.unsqueeze(0)

        for i in range(3):
            if edge_attr is None:
                m = torch.nn.functional.relu(self.conv(out, edge_index))
            else:
                m = torch.nn.functional.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        out = torch.nn.functional.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)