import pickle

import torch
from torch.nn import BCELoss, CrossEntropyLoss, Linear, MSELoss
from torch_geometric.datasets import MoleculeNet
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import is_undirected
from torch_sparse import transpose
from tqdm import tqdm

from _project.exp_data import getBinaryClassifier, getQM7b, getZINC, getPPI, getMNIST
from _project.exp_explain import pick_explainer, generate_roar_training_data, generate_gsat_roar_training_data
from _project.exp_train import trainAndValidate, compare_orig_roar, trainAndValidateGSAT, compare_GSAT_orig_roar
from example.gsat import GSAT
from example.trainer import run_one_epoch
from exp_models import Model_BinClassifier, Model_Regressor, Model_PPI
from GSAT.src.utils import get_model
from GSAT.src.run_gsat import ExtractorMLP

from GSAT.src.utils import Criterion
from utils import reorder_like, process_data


class model_wrapper(torch.nn.Module):
    def __init__(self, orig_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = orig_model
        self.training = False

        self.clf = self.model.clf
        self.extractor = self.model.extractor
        self.sampling = self.model.sampling
        self.learn_edge_att = self.model.learn_edge_att
        self.lift_node_att_to_edge_att = self.model.lift_node_att_to_edge_att
        self.__loss__ = self.model.__loss__
        self.mapper = Linear(1, 2)

    def forward(self, x, edge_index, fulldata):
        # att, loss, loss_dict, clf_logits = self.model.forward_pass(fulldata.to(device), 0, training=False)
        training = True
        epoch = 0
        self.clf.training = True

        emb = self.clf.get_emb(fulldata.x, fulldata.edge_index, batch=fulldata.batch, edge_attr=fulldata.edge_attr)
        att_log_logits = self.extractor(emb, fulldata.edge_index, fulldata.batch)
        att = self.sampling(att_log_logits, training)

        if self.learn_edge_att:
            if is_undirected(fulldata.edge_index):
                trans_idx, trans_val = transpose(fulldata.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, fulldata.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, fulldata.edge_index)

        clf_logits = self.clf(fulldata.x, fulldata.edge_index, fulldata.batch, edge_attr=fulldata.edge_attr,
                              edge_atten=edge_att)
        #loss, loss_dict = self.__loss__(att, clf_logits, fulldata.y, epoch)
        # return edge_att, loss, loss_dict, clf_logits
        # clf_logits = clf_logits.flatten()
        # bin_ret = torch.nn.functional.one_hot(torch.where(clf_logits > 0.5, 1, 0), 2)
        # bin_ret = torch.where(bin_ret == 1, clf_logits.item(), 1-clf_logits.item())
        self.mapper = self.mapper.to(clf_logits.device)
        return self.mapper(clf_logits)

        return clf_logits



def create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device):
    model_clf = get_model(x_dim=num_features, edge_attr_dim=n_edge_attr, num_class=num_classes,
                          multi_label=is_multilabel, model_config=model_config, device=device)
    extractor = ExtractorMLP(model_config['hidden_size'],
                             shared_config={'learn_edge_att': True, 'extractor_dropout_p': 0.1}).to(device)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model_clf.parameters()), lr=1e-3,
                                 weight_decay=3.0e-6)
    criterion = Criterion(num_classes, is_multilabel)
    model = GSAT(model_clf, extractor, criterion, optimizer, learn_edge_att=False, final_r=0.7)

    return model

def run_experiments(dl="binary", TOPK=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    batch_size = 32
    #num_epochs = 20
    num_epochs = 25
    #roar_epochs = 5
    roar_epochs = 25

    orig_lr = 0.001
    roar_lr = 0.001


    if dl == "binary":
        train_dl, val_dl, test_dl, num_features, num_classes = getBinaryClassifier(batch_size)
        use_edge_attr = True
        n_edge_attr = train_dl.dataset[0].edge_attr.shape[-1]
        model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': use_edge_attr}
        is_multilabel = False

        orig_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)
        roar_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)

        y_fmt = "argmax"
        mode_type = 'multiclass_classification'
        return_type = 'log_probs'
        y_type = torch.long
        roar_loss_fn = MSELoss()
        mod_y = True

    if dl == "zinc":
        train_dl, val_dl, test_dl, num_features, num_classes = getZINC(batch_size)
        use_edge_attr = True
        n_edge_attr = train_dl.dataset[0].edge_attr.shape[-1]
        model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': use_edge_attr}
        is_multilabel = False
        mod_y = False
        num_classes = 1
        orig_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)
        roar_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)

        y_fmt = "none"
        mode_type = 'regression'
        return_type = 'raw'
        y_type = torch.float32
        roar_loss_fn = MSELoss()
        mod_y = False

    if dl == "ppi":
        train_dl, val_dl, test_dl, num_features, num_classes = getPPI(2)
        use_edge_attr = True
        n_edge_attr = train_dl.dataset[0].edge_attr.shape[-1]
        model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': use_edge_attr}
        is_multilabel = False

        orig_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)
        roar_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)

        y_fmt = "none"
        mode_type = 'multiclass_classification'
        return_type = 'probs'
        y_type = torch.long
        roar_loss_fn = CrossEntropyLoss()
        mod_y = False

    if dl == "mnist":
        train_dl, val_dl, test_dl, num_features, num_classes = getMNIST(128)
        use_edge_attr = True
        n_edge_attr = train_dl.dataset[0].edge_attr.shape[-1]
        model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': use_edge_attr}
        is_multilabel = False

        orig_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)
        roar_model = create_gsat_model(num_features, n_edge_attr, num_classes, is_multilabel, model_config, device)

        y_fmt = "none"
        mode_type = 'multiclass_classification'
        return_type = 'log_probs'
        y_type = torch.long
        roar_loss_fn = CrossEntropyLoss()
        mod_y = False


    orig_model = orig_model.to(device)
    orig_model = trainAndValidateGSAT(orig_model, train_dl, val_dl, num_epochs, use_edge_attr, device, mod_y)

    exp_type = "gnn"

    print(f"starting ROAR for: {exp_type}")

    #wrapped_model = model_wrapper(orig_model)
    wrapped_model = orig_model.clf
    explainer = pick_explainer(exp_type, wrapped_model, topk=TOPK, mode_type=mode_type, return_type=return_type)

    roar_training_data = generate_gsat_roar_training_data(val_dl, explainer, device, use_edge_attr)
    fopen = open(f"test_{dl}_roar_training_data_{exp_type}.pkl", "wb")
    pickle.dump(roar_training_data, fopen)
    fopen.close()

    fopen = open(f"test_{dl}_roar_training_data_{exp_type}.pkl", "rb")
    roar_training_data = pickle.load(fopen)
    fopen.close()

    #roar_model = trainAndValidateGSAT(roar_model, roar_training_data, None, roar_epochs, use_edge_attr, device, mod_y=False)
    for epoch in range(1, roar_epochs+1):
        epoch_loss = 0
        val_epoch_loss = 0

        roar_model.clf.train()
        for data in tqdm(roar_training_data, unit="batch", total=len(roar_training_data)):
            data = process_data(data, use_edge_attr)
            data.edge_attr = data.edge_attr.to(torch.float32)
            data = data.to(device)
            roar_model.optimizer.zero_grad()

            try:
                clf_logits = roar_model.clf(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

                if mod_y:
                    data.y = data.y.argmax(-1).unsqueeze(dim=-1)
                loss = roar_loss_fn(clf_logits, data.target)
                loss.backward()
                roar_model.optimizer.step()

                epoch_loss += loss.item()
            except Exception as e:
                print("Error in training")
                pass


    compare_GSAT_orig_roar(orig_model.clf, roar_model.clf, test_dl, device, roar_loss_fn, y_fmt, y_type, use_edge_attr)


#run_experiments("binary", TOPK=2)
#run_experiments("mnist", TOPK=7)
run_experiments("zinc", TOPK=2)
#run_experiments("ppi", TOPK=120)