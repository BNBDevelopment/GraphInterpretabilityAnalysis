import os.path
import os.path as osp
from pathlib import Path

import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric import seed_everything

from torch_geometric.datasets import Planetoid, QM7b, PPI, GNNBenchmarkDataset, MNISTSuperpixels
from torch_geometric.explain import Explainer, GraphMaskExplainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv

from _project.explain_data import get_dataset
from _project.explain_models import GCN, GPS_Model
from _project.explain_train import train
from _project.explani_gen_explanations import gen_explanations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')

seed_everything(seed=12345)
lr = 0.00002
weight_decay=5e-4
b_size = 64

model_name = "GPS01"
ds_choice = 'mnist'
model_type = "GPS"

train_dl, val_dl, test_dl, n_features, n_classes = get_dataset(path, ds_choice, device, b_size)


# GCN Node Classification =====================================================

configuration = {
    'n_epochs': 100,
    #'optimizer': optimizer,
    'train_data': train_dl,
    'val_data': val_dl,
    'test_data': test_dl,
    'loss_fn': torch.nn.NLLLoss(),
    #'loss_fn': torch.nn.CrossEntropyLoss()
    'model_name': model_name,
    'model_save_path': "saved_models/",
    'save_strategy': "max_val_acc",
    'load_model': False,
    'model_hdim': 64,
    'model_n_layers': 3,
    'model_n_heads': 8,
    'n_features': n_features,
    'n_classes': n_classes,
    'roar_lr': 0.01,
    'roar_epochs': 5,
}

#







if model_name in [x[:len(model_name)] for x in os.listdir(configuration['model_save_path'])]:
    #model_name = "M1_Epoch_94_ValAcc_0.32083333333333336.pt"
    list_of_possible_files = [x for x in os.listdir(configuration['model_save_path']) if x.startswith(model_name)]
    assert len(list_of_possible_files) < 2, f"Error: too many files with prefix {model_name}. Unknown which model to load."

    model_save_path = configuration['model_save_path'] + list_of_possible_files[0]
    model = torch.load(model_save_path)
else:
    match model_type:
        case "GPS":
            model = GPS_Model(n_features, n_classes, h_dim=configuration['model_hdim'], n_layers=configuration['model_n_layers'], n_heads=configuration['model_n_heads']).to(device)
        case "GCN":
            model = GCN(n_features, n_classes, h_dim=128).to(device)
        case _:
            raise NotImplementedError(f"Model type '{model_type}' not valid!")

    configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, train_n_correct, val_loss, val_n_correct = train(model, configuration)





#test(model, configuration)

data_to_explain = configuration['val_data']
configuration['explanation_dir'] = "explanations/"

# original_graph = torch_geometric.utils.to_networkx(data_to_explain.dataset.data, to_undirected=True)
# nx.draw(original_graph, node_size=2.0)
# plt.savefig(configuration['explanation_dir'] + "original_graph.png")
#plt.show(block=False)

#gen_explanations("graph_mask", model, configuration, data_to_explain, n_explanations=len(data_to_explain.dataset), roar_test_data=test_dl)


for et in ['gnn_explain', 'attn_explain']:
    gen_explanations(et, model, configuration, data_to_explain, n_explanations=len(data_to_explain.dataset), roar_test_data=test_dl)

