import copy
import os
import pickle
import shutil

import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.explain import Explainer, GraphMaskExplainer, GNNExplainer, AttentionExplainer
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_mean_pool
from tqdm import tqdm

from _project.explain_models import GPS_Model, TransformerConv2


def get_explainer(method_type, model, configuration):
    if method_type == "graph_mask":
        explainer = Explainer(
            model=model,
            algorithm=GraphMaskExplainer(num_layers=configuration['model_n_layers'], epochs=5),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
            threshold_config=dict(threshold_type='topk', value=5)
        )
    elif method_type == "gnn_explain":
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=5),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
            threshold_config=dict(threshold_type='topk', value=5)
        )
    elif method_type == "attn_explain":
        explainer = Explainer(
            model=model,
            algorithm=AttentionExplainer(),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
            threshold_config=dict(threshold_type='topk', value=5)
        )
    else:
        raise NotImplementedError(f"Bad methodtype: {method_type}")

    return explainer




def gen_explanations(method_type, model, configuration, data_to_explain, n_explanations=1, roar_test_data=None, image_every=200, force_createData=False):
    # data = data_to_explain.dataset.data#configuration['data']
    model.eval()

    device = next(model.parameters()).device
    if os.path.isfile(f"{type(model).__name__}_{method_type}_{configuration['dataset_name']}_ALL_DATA.pt") and not force_createData:
        load_file = open(f"{type(model).__name__}_{method_type}_{configuration['dataset_name']}_ALL_DATA.pt", "rb")
        loaded_data = pickle.load(load_file)
        load_file.close()
        roar_train_samples = []
        node_importances = []
        edge_importances = []

        for v in loaded_data:
            roar_train_samples.append(v['subgraph'])
            if 'node_importances' in v.keys():
                node_importances.append(v['node_importances'])
            if 'edge_importances' in v.keys():
                node_importances.append(v['edge_importances'])

    else:
        roar_train_samples = []

        img_save_folder = configuration['explanation_dir'] + f"{configuration['model_name']}/"
        if not os.path.exists(img_save_folder):
            os.mkdir(img_save_folder)
        else:
            shutil.rmtree(img_save_folder, ignore_errors=False, onerror=None)
            os.mkdir(img_save_folder)

        explainer = get_explainer(method_type, model, configuration)

        data_to_explain = DataLoader(data_to_explain.dataset, batch_size=1, shuffle=False)

        #for idx_to_explain in range(n_explanations):
        idx_to_explain = 0
        for data in tqdm(data_to_explain, unit="batch", total=len(data_to_explain)):
            #data = data_to_explain.dataset[idx_to_explain]
            idx_to_explain += 1

            #transform_subsetFromMask = torch_geometric.transforms.Compose([])
            explanation = explainer(x=data.x, edge_index=data.edge_index, batch_mapping=data.batch)
            sub_graph = explanation.get_explanation_subgraph()
            sub_graph.y = data.y
            datapoint = {'subgraph': sub_graph}

            if 'node_mask' in explanation.available_explanations:
                # binary_node_mask = explanation.node_mask.squeeze() != 0
                # sub_graph = explanation.subgraph(binary_node_mask)
                datapoint['node_importances'] = explanation.node_masks
                roar_train_samples.append(datapoint)
            if 'edge_mask' in explanation.available_explanations:
                #binary_edge_mask = explanation.edge_mask.squeeze() != 0
                datapoint['edge_importances'] = explanation.edge_mask
            roar_train_samples.append(datapoint)

            if idx_to_explain % image_every == 0:
                #print(f"At iter {idx_to_explain}/{n_explanations}")
                filesave_name = f"{type(model).__name__}_{method_type}_item{idx_to_explain}_actual{data.y.item()}.png"
                explanation.visualize_graph(img_save_folder + filesave_name)

        save_file = open(f"{type(model).__name__}_{method_type}_{configuration['dataset_name']}_ALL_DATA.pt", "wb")
        pickle.dump(roar_train_samples, save_file)
        save_file.close()

        # convert to list format
        roar_train_samples = [x['subgraph'] for x in roar_train_samples]

    #Now Do ROAR strategy
    if not roar_test_data is None:
        doROAR(model, configuration, roar_train_samples, roar_test_data)
        #plt.show(block=False)




def doROAR(model, configuration, roar_train_data, roar_test_data):
    model.eval()
    num_classes = configuration['n_classes']
    in_channels = configuration['n_features']
    roar_epochs = configuration['roar_epochs']
    roar_lr = configuration['roar_lr']

    #mlp = MLP(in_channels=in_channels, hidden_channels=256, out_channels=num_classes, num_layers=3)


    #device = next(model.parameters()).device
    device = torch.device("cpu")
    loss_fn = configuration['loss_fn']
    do_embedding = configuration['do_embedding']
    inc = configuration['is_node_classification']
    #mlp = mlp.to(device)
    mlp = TransformerConv2(in_channels, num_classes, h_dim=32, n_layers=1, n_heads=2, do_embedding=do_embedding, is_node_classification=inc).to(device)
    mlp = mlp.to(device)


    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(roar_epochs):
        epochloss = 0
        for single_data_item in tqdm(roar_train_data, unit="batch", total=len(roar_train_data)):
            optimizer.zero_grad()
            x = single_data_item.x.to(device)
            e = single_data_item.edge_index.to(device)
            artificial_batch_index = torch.zeros(x.shape[0]).to(torch.int64).to(device)

            out = mlp(x=x, edge_index=e, batch_mapping=artificial_batch_index)
            #out = global_mean_pool(out, batch=None)
            #out = torch.nn.functional.softmax(out, dim=1)

            y = torch.nn.functional.one_hot(single_data_item.target, num_classes=num_classes).to(torch.float).to(device)
            loss = loss_fn(input=out, target=y)
            loss.backward()
            optimizer.step()
            epochloss += loss.item()
        print(f"ROAR average epoch {epoch} loss: {epochloss/len(roar_train_data)}")
    model.eval()
    correct = 0
    total = 0
    test_mse = 0
    for test_batch in roar_test_data:
        test_batch = test_batch.to(device)
        pred = mlp(x=test_batch.x, edge_index=test_batch.edge_index, batch_mapping=test_batch.batch)

        total += test_batch.y.shape[0]
        if configuration['n_classes'] == 1:
            test_mse += sum(pred).item()
        else:
            pred = pred.argmax(-1)
            correct += (pred == test_batch.y.argmax(-1)).sum().item()

    if configuration['n_classes'] == 1:
        mse = test_mse / total
        print(f'ROAR Test MSE: {mse:.4f}')
    else:
        acc = correct / total
        print(f'ROAR Test Accuracy: {acc:.4f}')