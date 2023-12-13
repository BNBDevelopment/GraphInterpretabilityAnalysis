import pickle

import torch
from torch.nn import BCELoss, CrossEntropyLoss
from torch_geometric.datasets import MoleculeNet
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from tqdm import tqdm

from _project.exp_data import getBinaryClassifier, getQM7b, getZINC, getPPI, getMNIST
from _project.exp_explain import pick_explainer, generate_roar_training_data
from _project.exp_train import trainAndValidate, compare_orig_roar, trainAndValidateGSAT
from example.gsat import GSAT
from example.trainer import run_one_epoch
from exp_models import Model_BinClassifier, Model_Regressor, Model_PPI
from GSAT.src.utils import get_model
from GSAT.src.run_gsat import ExtractorMLP

from GSAT.src.utils import Criterion


def run_experiments(dl="binary", TOPK=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    #num_epochs = 20
    num_epochs = 1
    #roar_epochs = 5
    roar_epochs = 1

    orig_lr = 0.001
    roar_lr = 0.001


    if dl == "binary":
        train_dl, val_dl, test_dl, num_features, num_classes = getBinaryClassifier(batch_size)
        use_edge_attr = True
        model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': use_edge_attr}
        is_multilabel = False
        clf = get_model(x_dim=64, edge_attr_dim=1, num_class=num_classes, multi_label=is_multilabel, model_config=model_config, device=device)
        extractor = ExtractorMLP(model_config['hidden_size'], shared_config={'learn_edge_att':True, 'extractor_dropout_p':0.1}).to(device)
        optimizer = torch.optim.Adam(list(extractor.parameters()) + list(clf.parameters()), lr=1e-3,
                                     weight_decay=3.0e-6)
        criterion = Criterion(num_classes, is_multilabel)
        gsat = GSAT(clf, extractor, criterion, optimizer, learn_edge_att=False, final_r=0.7)

        # model = Model_BinClassifier(num_features, num_classes, hdim=64).to(device)
        # roar_model_class = Model_BinClassifier
        loss_fn = torch.nn.functional.nll_loss
        y_fmt = "argmax"
        y_type = torch.long
    if dl == "zinc":
        train_dl, val_dl, test_dl, num_features, num_classes = getZINC(batch_size)
        model = Model_Regressor(num_features, num_classes, hdim=64).to(device)
        roar_model_class = Model_Regressor
        loss_fn = torch.nn.functional.mse_loss
        y_fmt = "none"
        y_type = torch.float32
    if dl == "ppi":
        train_dl, val_dl, test_dl, num_features, num_classes = getPPI(2)
        model = Model_PPI(num_features, num_classes, hdim=64).to(device)
        roar_model_class = Model_PPI
        loss_fn = CrossEntropyLoss()
        y_fmt = "none"
        y_type = torch.long
    if dl == "mnist":
        train_dl, val_dl, test_dl, num_features, num_classes = getMNIST(128)
        model = Model_BinClassifier(num_features, num_classes, hdim=64).to(device)
        roar_model_class = Model_BinClassifier
        loss_fn = CrossEntropyLoss()
        y_fmt = "none"
        y_type = torch.long



    #optimizer = torch.optim.Adam(model.parameters(), lr=orig_lr, weight_decay=5e-4)

    #train_res = run_one_epoch(gsat, loaders['train'], epoch, 'train', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'])

    #model = trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device, loss_fn, y_fmt, y_type=y_type)
    model = trainAndValidateGSAT(model, train_dl, val_dl, num_epochs, use_edge_attr)

    for exp_type in ["gsat"]:
        print(f"starting ROAR for: {exp_type}")

        roar_training_data = []
        for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
            data = data.to(device)
            try:
                explanation = explainer(data.x, data.edge_index)
                sub_graph = explanation.get_explanation_subgraph()
                roar_training_data.append(sub_graph)
            except:
                print("Error generating explanation")
                pass
        return roar_training_data

        roar_model = roar_model_class(num_features, num_classes, hdim=64).to(device)
        roar_optimizer = torch.optim.Adam(roar_model.parameters(), lr=roar_lr, weight_decay=5e-4)

        roar_model = trainAndValidate(roar_model, roar_training_data, None, roar_epochs, roar_optimizer, device, loss_fn, y_fmt=y_fmt, y_type=y_type)

        compare_orig_roar(model, roar_model, test_dl, device, loss_fn, y_fmt, y_type)


run_experiments("binary", TOPK=2)
#run_experiments("mnist", TOPK=7)
#run_experiments("zinc", TOPK=2)
#run_experiments("ppi", TOPK=120)