import math

import torch
from sklearn.metrics import f1_score
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer
from tqdm import tqdm


def calc_loss(data, out, loss_fn, y_fmt, y_type):
    if len(out.shape) == 3:
        out = out.transpose(1, 2).to(torch.float32)

    if data.y is None:
        if y_fmt == "argmax":
            loss = loss_fn(out, data.target.to(y_type))
        else:
            loss = loss_fn(out, data.target.to(y_type))
    else:
        if y_fmt == "argmax":
            loss = loss_fn(out, data.y.argmax(-1).to(y_type))
        else:
            loss = loss_fn(out, data.y.to(y_type))
    return loss


def count_correct(data, y, raw_prediction, y_fmt):
    if y_fmt == 'argmax':
        y = y.argmax(-1)

    if len(y.shape) > 1 and y.shape[1] > 1:
        #raw_prediction = raw_prediction.transpose(1,2)
        where_matches = raw_prediction.argmax(-1) == y
        matches = where_matches.sum().item()
        return matches, math.prod(list(y.shape))
    else:
       # actual = y.argmax(-1)
        if raw_prediction.argmax(-1) == y:
            return 1, 1
        else:
            return 0, 1

def bgnn_trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device, loss_fn=torch.nn.functional.nll_loss, y_fmt="argmax", y_type=torch.float32, mod_e=True, do_bn=True):
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        for data in tqdm(train_dl, unit="batch", total=len(train_dl)):
            data = data.to(device)
            optimizer.zero_grad()

            if mod_e:
                data.edge_attr = data.edge_attr[:,0].to(torch.float32)

            try:
                out = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos, do_bn)
                loss = calc_loss(data, out, loss_fn, y_fmt, y_type=y_type)
                loss.backward()
                optimizer.step()
                if not torch.isnan(loss):
                    epoch_loss += loss.item()
            except Exception as e:
                print("Error in training")
                pass

        if not val_dl is None:
            val_loss = 0
            val_total = 0
            val_correct = 0
            val_ys = []
            val_preds = []
            score_model = 0

            model.eval()
            for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
                data = data.to(device)
                data.edge_attr = data.edge_attr[:, 0].to(torch.float32)
                try:
                    out = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                    #loss = loss_fn(out, data.y.argmax(-1))
                    loss = calc_loss(data, out, loss_fn, y_fmt, y_type=y_type)
                    val_epoch_loss += loss.item()

                    plus_val, plus_total = count_correct(data, data.y, raw_prediction=out, y_fmt=y_fmt)
                    val_total += plus_total
                    val_correct += plus_val

                    val_pr = out.argmax(-1)
                    val_ys.append(data.y.flatten().cpu())
                    val_preds.append(val_pr.flatten().cpu())
                except Exception as e:
                    print("Error vlaidation")
                    pass

                if val_total > 200:
                    break


            try:
                f1_y = torch.cat(val_ys, dim=0).numpy()
                f1_model = torch.cat(val_preds, dim=0).numpy()
                score_model = f1_score(f1_y, f1_model, average='micro')
            except:
                pass


            print(f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss/len(train_dl)} val loss: {val_epoch_loss/val_total}  val_acc:{val_correct/val_total}  val f1: {score_model}")
        else:
            print(
                f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss / len(train_dl)}")

    return model


def generate_bgnn_roar_training_data(val_dl, explainer, device):
    roar_training_data = []
    for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
        data = data.to(device)
        data.edge_attr = data.edge_attr[:, 0].to(torch.float32)
        # try:
        explanation = explainer(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, pos=data.pos)
        sub_graph = explanation.get_explanation_subgraph()
        roar_training_data.append(sub_graph)
        # except Exception as e:
        #     print("Error generating explanation")
        #     pass
    return roar_training_data

def compare_gnn_orig_roar(model, roar_model, test_dl, device, loss_fn, y_fmt, y_type):
    roar_model.eval()
    model.eval()

    orig_model_correct = 0
    roar_model_correct = 0
    o_total = 0
    r_total = 0
    orig_model_loss = 0
    roar_model_loss = 0
    total = 0

    ys = []
    model_preds = []
    roar_preds = []


    for data in tqdm(test_dl, unit="batch", total=len(test_dl)):
        data = data.to(device)
        data.edge_attr = data.edge_attr[:, 0].to(torch.float32)
        modl_out = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        roar_out = roar_model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)

        total += data.y.shape[0]
        orig_new_correct, new_o_total = count_correct(data, data.y, raw_prediction=modl_out, y_fmt=y_fmt)
        roar_new_correct, new_r_total = count_correct(data, data.y, raw_prediction=roar_out, y_fmt=y_fmt)
        orig_model_correct += orig_new_correct
        roar_model_correct += roar_new_correct
        o_total += new_o_total
        r_total += new_r_total


        # if y_fmt == "argmax":
        #     if data.y is None:
        #         y_final = data.target
        #     else:
        #         y_final = data.y.argmax(-1)
        # else:
        #     y_final = data.y
        # model_loss = loss_fn(modl_out, y_final)
        # roar_loss = loss_fn(roar_out, y_final)

        model_loss = calc_loss(data, modl_out, loss_fn, y_fmt, y_type=y_type)
        roar_loss = calc_loss(data, roar_out, loss_fn, y_fmt, y_type=y_type)

        orig_model_loss += model_loss.item()
        roar_model_loss += roar_loss.item()

        modl_pred = modl_out.argmax(-1)
        roar_pred = roar_out.argmax(-1)
        ys.append(data.y.flatten().cpu())
        model_preds.append(modl_pred.flatten().cpu())
        roar_preds.append(roar_pred.flatten().cpu())


    print(f"Original Model accuracy: {orig_model_correct / o_total} \t\t Roar Model accuracy: {roar_model_correct / r_total}")
    print(f"Original Model loss: {orig_model_loss / total} \t\t Roar Model loss: {roar_model_loss / total}")

    try:
        f1_y = torch.cat(ys, dim=0).numpy()
        f1_model = torch.cat(model_preds, dim=0).numpy()
        f1_roar = torch.cat(roar_preds, dim=0).numpy()
        score_model = f1_score(f1_y, f1_model, average='micro')
        score_roar = f1_score(f1_y, f1_roar, average='micro')

        print(f"Original Model F1: {score_model} \t\t Roar Model F1: {score_roar}")
    except:
        print("error calculating final f1")
        pass


def pick_bgnn_explainer(name, model, topk=3, mode_type='multiclass_classification', return_type='log_probs', edge_mask_type=None):

    if name == "gnn":
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=15),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=edge_mask_type,
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
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode=mode_type,
                task_level='graph',
                return_type=return_type,
            ),
            threshold_config=dict(threshold_type='topk', value=topk)
        )

    return explainer