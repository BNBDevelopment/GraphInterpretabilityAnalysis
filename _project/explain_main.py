import os.path as osp

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

from _project.explain_models import GCN, GPS_Model
from _project.explain_train import train
from _project.explani_gen_explanations import gen_explanations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')

seed_everything(seed=12345)
lr = 0.00002
weight_decay=5e-4
b_size = 64

model_name = "M1"
ds_choice = 'mnist'


transform_datasplit = torch_geometric.transforms.RandomNodeSplit(num_val=0.1, num_test=0.1)
transform_splittrainval = torch_geometric.transforms.RandomLinkSplit(num_val=0.0, num_test=0.15)
transform_toDevice = torch_geometric.transforms.ToDevice(device=device)

train_percent = 0.8
val_percent = 0.1
test_percetn = 0.1

if ds_choice == 'benchmark':
    train_dataset = GNNBenchmarkDataset(path, name='MNIST', split='train')
    val_dataset = GNNBenchmarkDataset(path, name='MNIST', split='val')
    test_dataset = GNNBenchmarkDataset(path, name='MNIST', split='test')

elif ds_choice == 'mnist':
    train_dataset = MNISTSuperpixels(path, train=True)
    val_start = round(len(train_dataset) * (1 - val_percent))

    val_dataset = train_dataset[val_start:]
    train_dataset = train_dataset[:val_start]
    test_dataset = MNISTSuperpixels(path, train=False)

elif ds_choice == 'qm7b':
    train_dataset = QM7b(path)
elif ds_choice == 'ppi':
    train_dataset = PPI(path, split='train',)


train_dataset.data = transform_toDevice(train_dataset.data)
val_dataset.data = transform_toDevice(val_dataset.data)
test_dataset.data = transform_toDevice(test_dataset.data)


train_dl = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=b_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=b_size, shuffle=True)


data = train_dataset[0].to(device)
n_features = train_dataset.num_features
n_classes = train_dataset.num_classes



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
}

model = GCN(n_features, n_classes, h_dim=128).to(device)
#model = GPS_Model(n_features, n_classes, h_dim=configuration['model_hdim'], n_layers=configuration['model_n_layers'], n_heads=configuration['model_n_heads']).to(device)

configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)






if not (configuration['load_model']):
    train_loss, train_n_correct, val_loss, val_n_correct = train(model, configuration)
else:
    model_name = "M1_Epoch_94_ValAcc_0.32083333333333336.pt"
    model_save_path = configuration['model_save_path']
    model = torch.load(model_save_path + model_name)

#test(model, configuration)

data_to_explain = configuration['test_data']
configuration['explanation_dir'] = "explanations/"

# original_graph = torch_geometric.utils.to_networkx(data_to_explain.dataset.data, to_undirected=True)
# nx.draw(original_graph, node_size=2.0)
# plt.savefig(configuration['explanation_dir'] + "original_graph.png")
#plt.show(block=False)


gen_explanations("graph_mask", model, configuration, data_to_explain)

# GAT Node Classification =====================================================

# model = GAT(n_features, n_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
# for epoch in range(1, 201):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#
# explainer = Explainer(
#     model=model,
#     algorithm=GraphMaskExplainer(2, epochs=5),
#     explanation_type='model',
#     node_mask_type='attributes',
#     edge_mask_type='object',
#     model_config=dict(
#         mode='multiclass_classification',
#         task_level='node',
#         return_type='log_probs',
#     ),
# )
#
# node_index = torch.tensor([10, 20])
# explanation = explainer(data.x, data.edge_index, index=node_index)
# print(f'Generated explanations in {explanation.available_explanations}')