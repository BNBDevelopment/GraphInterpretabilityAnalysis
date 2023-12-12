import math

import torch
from tqdm import tqdm





def calc_loss(data, out, loss_fn, y_fmt, y_type):
    if len(out.shape) == 3:
        out = out.transpose(1, 2).to(torch.float32)

    if data.y is None:
        if y_fmt == "argmax":
            loss = loss_fn(out, data.target.argmax(-1).to(y_type))
        else:
            loss = loss_fn(out, data.target.to(y_type))
    else:
        if y_fmt == "argmax":
            loss = loss_fn(out, data.y.argmax(-1).to(y_type))
        else:
            loss = loss_fn(out, data.y.to(y_type))
    return loss



def count_correct(data, y, raw_prediction):
    if len(data.y.shape) > 1 and data.y.shape[1] > 1:
        #raw_prediction = raw_prediction.transpose(1,2)
        where_matches = raw_prediction.argmax(-1) == y
        matches = where_matches.sum().item()
        return matches, math.prod(list(y.shape))
    else:
        actual = y.argmax(-1)
        if raw_prediction == actual:
            return 1, 1
        else:
            return 0, 1




def trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device, loss_fn=torch.nn.functional.nll_loss, y_fmt="argmax", y_type=torch.float32):
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        for data in tqdm(train_dl, unit="batch", total=len(train_dl)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = calc_loss(data, out, loss_fn, y_fmt, y_type=y_type)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if not val_dl is None:
            val_loss = 0
            val_total = 0
            val_correct = 0
            model.eval()
            for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
                data = data.to(device)

                out = model(data.x, data.edge_index, data.batch)
                #loss = loss_fn(out, data.y.argmax(-1))
                loss = calc_loss(data, out, loss_fn, y_fmt, y_type=y_type)
                val_epoch_loss += loss.item()


                plus_val, plus_total = count_correct(data, data.y, raw_prediction=out)
                val_total += plus_total
                val_correct += plus_val


            print(f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss/len(train_dl)} val loss: {val_epoch_loss/len(val_dl)}  val_acc:{val_correct/val_total}")
        else:
            print(
                f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss / len(train_dl)}")

    return model

def compare_orig_roar(model, roar_model, test_dl, device, loss_fn, y_fmt, y_type):
    roar_model.eval()
    model.eval()

    orig_model_correct = 0
    roar_model_correct = 0
    o_total = 0
    r_total = 0
    orig_model_loss = 0
    roar_model_loss = 0
    total = 0

    for data in tqdm(test_dl, unit="batch", total=len(test_dl)):
        data = data.to(device)
        modl_out = model(data.x, data.edge_index, data.batch)
        roar_out = roar_model(data.x, data.edge_index, data.batch)

        # modl_pred = modl_out.argmax(-1)
        # roar_pred = roar_out.argmax(-1)

        total += data.y.shape[0]
        orig_new_correct, new_o_total = count_correct(data, data.y, raw_prediction=modl_out)
        roar_new_correct, new_r_total = count_correct(data, data.y, raw_prediction=roar_out)
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

    print(f"Original Model accuracy: {orig_model_correct / o_total} \t\t Roar Model accuracy: {roar_model_correct / r_total}")
    print(f"Original Model loss: {orig_model_loss / total} \t\t Roar Model loss: {roar_model_loss / total}")