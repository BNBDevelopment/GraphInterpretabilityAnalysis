import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from tqdm import tqdm

from GSAT.src.run_gsat import ExtractorMLP, GSAT
from GSAT.src.utils import Criterion
from _project.exp_data import getBinaryClassifier
from _project.exp_explain import pick_explainer, generate_roar_training_data
from _project.exp_train import trainAndValidate, compare_orig_roar


class Model01(torch.nn.Module):
    def __init__(self, num_features, num_classes, hdim=64, n_heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, hdim//n_heads, heads=n_heads)
        self.conv2 = GATConv(hdim, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if not batch is None:
            x = global_mean_pool(x, batch)
        return torch.nn.functional.log_softmax(x, dim=1)





def run_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 10
    roar_epochs = 3
    hidden_size = 64

    train_dl, val_dl, test_dl, num_features, num_classes = getBinaryClassifier(batch_size)


    model = Model01(num_features, num_classes, hdim=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model = trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device)

    use_multilabel = False

    extractor = ExtractorMLP(hidden_size, learn_edge_att=False).to(device)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=1e-3, weight_decay=3.0e-6)
    criterion = Criterion(num_classes, use_multilabel)
    gsat = GSAT(model, extractor, criterion, optimizer, learn_edge_att=False, final_r=0.7)

    for exp_type in ["gnn", "atn"]:
        print(f"starting ROAR for: {exp_type}")
        explainer = pick_explainer(exp_type, model, topk=3)

        roar_training_data = generate_roar_training_data(val_dl, explainer, device)

        roar_model = Model01(num_features, num_classes, hdim=64).to(device)
        roar_model = trainAndValidate(roar_model, roar_training_data, None, num_epochs, optimizer, device)

        compare_orig_roar(model, roar_model, test_dl, device)



run_experiments()