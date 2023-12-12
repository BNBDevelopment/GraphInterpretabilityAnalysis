import pickle

import torch
from torch.nn import BCELoss, CrossEntropyLoss
from torch_geometric.datasets import MoleculeNet
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from tqdm import tqdm

from _project.exp_data import getBinaryClassifier, getQM7b, getZINC, getPPI
from _project.exp_explain import pick_explainer, generate_roar_training_data
from _project.exp_train import trainAndValidate, compare_orig_roar
from exp_models import Model_BinClassifier, Model_Regressor, Model_PPI


def run_experiments(dl="binary"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    #num_epochs = 20
    num_epochs = 1
    #roar_epochs = 5
    roar_epochs = 1

    if dl == "binary":
        train_dl, val_dl, test_dl, num_features, num_classes = getBinaryClassifier(batch_size)
        model = Model_BinClassifier(num_features, num_classes, hdim=64).to(device)
        roar_model = Model_BinClassifier(num_features, num_classes, hdim=64).to(device)
        loss_fn = torch.nn.functional.nll_loss
        y_fmt = "argmax"
        mode_type = 'multiclass_classification'
        return_type = 'log_probs'
        y_type = torch.float32
    if dl == "zinc":
        train_dl, val_dl, test_dl, num_features, num_classes = getZINC(batch_size)
        model = Model_Regressor(num_features, num_classes, hdim=64).to(device)
        roar_model = Model_Regressor(num_features, num_classes, hdim=64).to(device)
        loss_fn = torch.nn.functional.mse_loss
        y_fmt = "none"
        mode_type = 'regression'
        return_type = 'raw'
        y_type = torch.float32
    if dl == "ppi":
        train_dl, val_dl, test_dl, num_features, num_classes = getPPI(2)
        model = Model_PPI(num_features, num_classes, hdim=64).to(device)
        roar_model = Model_PPI(num_features, num_classes, hdim=64).to(device)
        loss_fn = CrossEntropyLoss()
        y_fmt = "none"
        mode_type = 'multiclass_classification'
        return_type = 'probs'
        y_type = torch.long



    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model = trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device, loss_fn, y_fmt, y_type=y_type)

    for exp_type in ["gnn", "atn"]:
        print(f"starting ROAR for: {exp_type}")
        explainer = pick_explainer(exp_type, model, topk=3, mode_type=mode_type, return_type=return_type)

        roar_training_data = generate_roar_training_data(val_dl, explainer, device)
        fopen = open(f"{dl}_roar_training_data_{exp_type}.pkl", "wb")
        pickle.dump(roar_training_data, fopen)
        fopen.close()

        fopen = open(f"{dl}_roar_training_data_{exp_type}.pkl", "rb")
        roar_training_data = pickle.load(fopen)
        fopen.close()

        roar_model = trainAndValidate(roar_model, roar_training_data, None, num_epochs, optimizer, device, loss_fn, y_fmt=y_fmt, y_type=y_type)

        compare_orig_roar(model, roar_model, test_dl, device, loss_fn, y_fmt, y_type)



#run_experiments("binary")
#run_experiments("zinc")
run_experiments("ppi")