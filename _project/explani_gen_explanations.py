import copy

import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.explain import Explainer, GraphMaskExplainer, GNNExplainer
import networkx as nx

def gen_explanations(method_type, model, configuration, data_to_explain, n_explanations=1):
    # data = data_to_explain.dataset.data#configuration['data']
    count = 0

    for i in range(n_explanations):
        data = data_to_explain.dataset[i]
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
                threshold_config=dict(threshold_type='topk', value=1)
            )

        #transform_subsetFromMask = torch_geometric.transforms.Compose([])
        explanation = explainer(x=data.x, edge_index=data.edge_index, batch_mapping=data.batch)

        binary_node_mask = explanation.node_mask.squeeze() != 0
        binary_edge_mask = explanation.edge_mask.squeeze() != 0

        #new_x = torch_geometric.utils.mask_select(src=data.x, dim=0, mask=binary_node_mask)
        #new_e = torch_geometric.utils.mask_select(src=data.edge_index, dim=0, mask=explanation.edge_mask)

        #explanation_data = copy.deepcopy(data)
        #explanation_data.x = new_x
        #explanation_data.e = new_e
        explanation.to('cpu')

        filesave_name = type(model).__name__ + "_" + method_type + ".png"
        explanation_graph = torch_geometric.utils.to_networkx(explanation, to_undirected=True)
        nx.draw(explanation_graph, node_size=50.0, node_color=explanation.node_mask.squeeze())
        plt.savefig(configuration['explanation_dir'] + filesave_name, format="PNG")
    #plt.show(block=False)