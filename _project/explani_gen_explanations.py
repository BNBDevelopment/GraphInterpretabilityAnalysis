import copy
import os
import pickle
import shutil

import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.explain import Explainer, GraphMaskExplainer, GNNExplainer, AttentionExplainer
import networkx as nx
from torch_geometric.nn import MLP, global_mean_pool
from tqdm import tqdm

from _project.explain_models import GPS_Model


def gen_explanations(method_type, model, configuration, data_to_explain, n_explanations=1, roar_test_data=None, image_every=100, gen_exp_dataset=True):
    # data = data_to_explain.dataset.data#configuration['data']

    if gen_exp_dataset:
        only_important_data = {'x':[],
                              'y':[],
                              'model_preds':[]}
        only_important_dataset = []

        img_save_folder = configuration['explanation_dir'] + f"{configuration['model_name']}/"
        if not os.path.exists(img_save_folder):
            os.mkdir(img_save_folder)
        else:
            shutil.rmtree(img_save_folder, ignore_errors=False, onerror=None)
            os.mkdir(img_save_folder)

        for idx_to_explain in range(n_explanations):
        #for data in tqdm(data_to_explain, unit="batch", total=len(data_to_explain)):
            data = data_to_explain.dataset[idx_to_explain]
            explainer = None

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

            #transform_subsetFromMask = torch_geometric.transforms.Compose([])
            explanation = explainer(x=data.x, edge_index=data.edge_index, batch_mapping=data.batch)

            binary_node_mask = explanation.node_mask.squeeze() != 0
            binary_edge_mask = explanation.edge_mask.squeeze() != 0

            new_x = torch_geometric.utils.mask_select(src=data.x, dim=0, mask=binary_node_mask)
            new_pos = torch_geometric.utils.mask_select(src=data.pos, dim=0, mask=binary_node_mask)
            new_e = torch_geometric.utils.mask_select(src=data.edge_index, dim=1, mask=binary_edge_mask)
            new_ea = None
            if not data.edge_attr is None:
                new_ea = torch_geometric.utils.mask_select(src=data.edge_attr, dim=1, mask=binary_edge_mask)

            important_only_samples = torch_geometric.data.Data(x=new_x, edge_index=new_e, edge_attr=new_ea, y=data.y, pos=new_pos)
            only_important_dataset.append(important_only_samples)


            explanation.to('cpu')

            artificial_batch_index = torch.zeros(data.x.shape[0]).to(torch.int64).to(data.x.device)
            model_log_pred = model(x=data.x, edge_index=data.edge_index, batch_mapping=artificial_batch_index)
            model_pred = torch.argmax(model_log_pred, dim=-1).item()
            actual_label = data.y.item()

            only_important_data['x'].append(data.x.detach())
            only_important_data['y'].append(data.y.detach())
            only_important_data['model_preds'].append(model_pred)



            if idx_to_explain % image_every == 0:
                print(f"At iter {idx_to_explain}/{n_explanations}")
                filesave_name = f"{type(model).__name__}_{method_type}_item{idx_to_explain}_pred{model_pred}_actual{actual_label}.png"
                explanation_graph = torch_geometric.utils.to_networkx(explanation, to_undirected=True)
                nx.draw(explanation_graph, node_size=50.0, node_color=explanation.node_mask.squeeze())
                plt.savefig(img_save_folder + filesave_name, format="PNG")

        save_file = open(f"{type(model).__name__}_{method_type}_ALL_DATA.pt", "wb")
        pickle.dump(only_important_dataset, save_file)
        save_file.close()
    else:
        load_file = open(f"{type(model).__name__}_{method_type}_ALL_DATA.pt", "rb")
        important_only_samples = pickle.load(load_file)
        load_file.close()

    #Now Do ROAR strategy
    if not roar_test_data is None:
        doROAR(model, configuration, important_only_samples, roar_test_data)
        #plt.show(block=False)

def doROAR(model, configuration, roar_train_data, roar_test_data):
    num_classes = configuration['n_classes']
    in_channels = configuration['n_features']
    roar_epochs = configuration['roar_epochs']
    roar_lr = configuration['roar_lr']

    #mlp = MLP(in_channels=in_channels, hidden_channels=256, out_channels=num_classes, num_layers=3)


    #device = next(model.parameters()).device
    device = torch.device("cpu")
    #mlp = mlp.to(device)
    mlp = GPS_Model(in_channels, num_classes, h_dim=32, n_layers=2, n_heads=4).to(device)
    mlp = mlp.to(device)

    artificial_batch_index = torch.zeros(roar_train_data[0].x.shape[0]).to(torch.int64).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=5e-4)

    for epoch in range(roar_epochs):
        epochloss = 0
        for single_data_item in roar_train_data:
            optimizer.zero_grad()
            x = single_data_item.x.to(device)
            e = single_data_item.edge_index.to(device)
            out = mlp(x=x, edge_index=e, batch_mapping=artificial_batch_index)
            #out = global_mean_pool(out, batch=None)
            #out = torch.nn.functional.softmax(out, dim=1)

            y = torch.nn.functional.one_hot(single_data_item.y, num_classes=num_classes).to(torch.float).to(device)
            loss = torch.nn.functional.binary_cross_entropy(input=out, target=y)
            loss.backward()
            optimizer.step()
            epochloss += loss.item()
        print(f"average epochloss: {epochloss/len(roar_train_data)}")
    model.eval()
    correct = 0
    total = 0
    for test_batch in roar_test_data:
        test_batch = test_batch.to(device)
        pred = mlp(x=test_batch.x, batch=test_batch.batch, batch_size=64)
        pred = global_mean_pool(pred, batch=test_batch.batch)
        pred = torch.nn.functional.softmax(pred, dim=1).argmax(-1)

        correct += (pred == test_batch.y).sum().item()
        total += len(test_batch)

    acc = correct / total
    print(f'Accuracy: {acc:.4f}')