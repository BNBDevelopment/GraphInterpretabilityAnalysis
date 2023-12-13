import torch
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from tqdm import tqdm

from utils import process_data


def generate_roar_training_data(val_dl, explainer, device):
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

def pick_explainer(name, model, topk=3, mode_type='multiclass_classification', return_type='log_probs'):

    if name == "gnn":
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=15),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode=mode_type,
                task_level='graph',
                return_type=return_type,
            ),
            threshold_config=dict(threshold_type='topk', value=topk)
        )
    if name == "atn":
        explainer = Explainer(
            model=model,
            algorithm=AttentionExplainer(),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode=mode_type,
                task_level='graph',
                return_type=return_type,
            ),
            threshold_config=dict(threshold_type='topk', value=topk)
        )

    return explainer


def generate_gsat_roar_training_data(val_dl, explainer, device, use_edge_attr):
    roar_training_data = []
    for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
        data = data.to(device)
        data = process_data(data, use_edge_attr)
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.y = data.y.argmax(-1).unsqueeze(dim=-1)

        try:
            torch.set_grad_enabled(True)
            explanation = explainer(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            sub_graph = explanation.get_explanation_subgraph()
            roar_training_data.append(sub_graph)
        except:
            print("Failure in GSAT training data generation")
            pass

    return roar_training_data