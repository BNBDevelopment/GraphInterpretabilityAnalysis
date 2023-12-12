import os.path
import os.path as osp

import torch
from torch_geometric import seed_everything

from _project.old2.explain_data import get_dataset
from _project.old2.explain_models import GCN, GPS_Model, AttentionConv_Model, TransformerConv2, CloneNet
from _project.old2.explain_train import train, test
from _project.old2.explani_gen_explanations import gen_explanations


def train_new_model(model_type, configuration, n_features, n_classes, do_embedding, is_node_classification, lr):
    match model_type:
        case "GPS":
            model = GPS_Model(n_features, n_classes, h_dim=configuration['model_hdim'],
                              n_layers=configuration['model_n_layers'], n_heads=configuration['model_n_heads'],
                              do_embedding=do_embedding).to(device)
        case "ATTN":
            model = GPS_Model(n_features, n_classes, h_dim=configuration['model_hdim'],
                              n_layers=configuration['model_n_layers'], n_heads=configuration['model_n_heads'],
                              do_embedding=do_embedding).to(device)
        case "GCN":
            model = GCN(n_features, n_classes, h_dim=128, do_embedding=do_embedding).to(device)
        case "ATTNCONV":
            model = AttentionConv_Model(n_features, n_classes, h_dim=configuration['model_hdim'],
                                        n_layers=configuration['model_n_layers'],
                                        n_heads=configuration['model_n_heads'], do_embedding=do_embedding).to(device)
        case "TCONV2":
            model = TransformerConv2(n_features, n_classes, h_dim=configuration['model_hdim'],
                                     n_layers=configuration['model_n_layers'], n_heads=configuration['model_n_heads'],
                                     do_embedding=do_embedding, is_node_classification=is_node_classification).to(device)
        case "CloneNet":
            model = CloneNet(n_features, n_classes, h_dim=configuration['model_hdim'],
                                     n_layers=configuration['model_n_layers'], n_heads=configuration['model_n_heads'],
                                     do_embedding=do_embedding, is_node_classification=is_node_classification).to(device)
        case _:
            raise NotImplementedError(f"Model type '{model_type}' not valid!")

    configuration['optimizer'] = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss, train_n_correct, val_loss, val_n_correct = train(model, configuration)

    return model


def run_experiment(model_name, ds_choice, model_type, lr, weight_decay, b_size, device, config_overwrite=None):
    train_dl, val_dl, test_dl, n_features, n_classes, do_embedding, is_node_classification = get_dataset(path, ds_choice, device)

    if n_classes is None:
        print("Doing regression! Num targets = 1")
        n_classes = 1

    configuration = {
        'n_epochs': 50,
        'dataset_name': ds_choice,
        #'optimizer': optimizer,
        'train_data': train_dl,
        'val_data': val_dl,
        'test_data': test_dl,
        #'loss_fn': torch.nn.NLLLoss(),

        'loss_fn': torch.nn.CrossEntropyLoss(),
        'save_strategy': "max_val_acc",

        # 'loss_fn': torch.nn.MSELoss(),
        # 'save_strategy': "min_val_loss",

        'model_name': model_name,
        'model_save_path': "saved_models/",
        'load_model': False,
        'model_hdim': 128,
        'model_n_layers': 3,
        'model_n_heads': 8,
        'n_features': n_features,
        'n_classes': n_classes,
        'roar_lr': lr,
        'roar_epochs': 5,
        'do_embedding': do_embedding,
        'device': device,
        'is_node_classification': is_node_classification,
    }

    for k,v in config_overwrite.items():
        configuration[k] = v

    if model_name in [x[:len(model_name)] for x in os.listdir(configuration['model_save_path'])]:
        #model_name = "M1_Epoch_94_ValAcc_0.32083333333333336.pt"
        list_of_possible_files = [x for x in os.listdir(configuration['model_save_path']) if x.startswith(model_name)]
        ds_filtered = [x for x in list_of_possible_files if configuration['dataset_name'] in x]
        if len(ds_filtered) == 1:
            model_save_path = configuration['model_save_path'] + list_of_possible_files[0]
            print(f"Loading model at {model_save_path}")
            model = torch.load(model_save_path)
        elif len(ds_filtered) > 1:
            raise Exception(f"ERROR - Found too many models! {ds_filtered}")
        else:
            print(f"Training new model!")
            model = train_new_model(model_type, configuration, n_features, n_classes, do_embedding, is_node_classification, lr)
    else:
        print(f"Training new model!")
        model = train_new_model(model_type, configuration, n_features, n_classes, do_embedding, is_node_classification, lr)


    test(model, configuration)

    data_to_explain = configuration['val_data']
    configuration['explanation_dir'] = "explanations/"

    for et in ['attn_explain', 'gnn_explain']:
        gen_explanations(et, model, configuration, data_to_explain, n_explanations=len(data_to_explain.dataset), roar_test_data=test_dl)


######################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'Planetoid')

seed_everything(seed=12345)
lr = 0.00002
weight_decay=5e-4
b_size = 32

# model_name = "TAttnConv01"
# ds_choice = 'mnist'
# model_type = "TCONV2"

if __name__ == '__main__':
    name = 'TCONV2'
    model_name = name + "_Exp02"

    over01 = {'loss_fn': torch.nn.CrossEntropyLoss(), 'save_strategy': "max_val_acc",
                'model_hdim': 64,
                'model_n_layers': 2,
                'model_n_heads': 2,}
    lr = 0.001
    run_experiment(model_name, 'benchmark', name, lr, weight_decay, b_size, device, config_overwrite=over01)

    over02 = {'loss_fn': torch.nn.MSELoss(), 'save_strategy': "min_val_loss"}
    lr = 0.00002
    run_experiment(model_name, 'zinc', 'CloneNet', lr, weight_decay, b_size, device, config_overwrite=over02)

    over03 = {'loss_fn': torch.nn.CrossEntropyLoss(), 'save_strategy': "max_val_acc",}
    lr = 0.00002
    run_experiment(model_name, 'ppi', name, lr, weight_decay, b_size, device, config_overwrite=over03)

    # over02 = {'loss_fn': torch.nn.MSELoss(), 'save_strategy': "min_val_loss"}
    # lr = 0.0001
    # run_experiment(model_name, 'qm7b', name, lr, weight_decay, b_size, device, config_overwrite=over02)