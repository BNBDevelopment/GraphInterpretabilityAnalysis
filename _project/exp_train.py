import torch
from tqdm import tqdm


def trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device, loss_fn=torch.nn.functional.nll_loss, y_fmt="argmax"):
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        for data in tqdm(train_dl, unit="batch", total=len(train_dl)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            if y_fmt == "argmax":
                if data.y is None:
                    loss = loss_fn(out, data.target)
                else:
                    loss = loss_fn(out, data.y.argmax(-1))
            else:
                loss = loss_fn(out, data.y)
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
                try:
                    out = model(data.x, data.edge_index, data.batch)
                    loss = loss_fn(out, data.y.argmax(-1))
                    val_epoch_loss += loss.item()

                    val_total += data.y.shape[0]
                    if out.argmax(-1) == data.y.argmax(-1):
                        val_correct += 1
                except:
                    pass

            print(f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss/len(train_dl)} val loss: {val_epoch_loss/len(val_dl)}  val_acc:{val_correct/val_total}")
        else:
            print(
                f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss / len(train_dl)}")

    return model

def compare_orig_roar(model, roar_model, test_dl, device, loss_fn, y_fmt):
    roar_model.eval()
    model.eval()

    orig_model_correct = 0
    roar_model_correct = 0
    total = 0
    orig_model_loss = 0
    roar_model_loss = 0

    for data in tqdm(test_dl, unit="batch", total=len(test_dl)):
        data = data.to(device)
        modl_out = model(data.x, data.edge_index, data.batch)
        roar_out = roar_model(data.x, data.edge_index, data.batch)

        modl_pred = modl_out.argmax(-1)
        roar_pred = roar_out.argmax(-1)
        actual = data.y.argmax(-1)

        total += data.y.shape[0]
        if modl_pred == actual:
            orig_model_correct += 1
        if roar_pred == actual:
            roar_model_correct += 1

        if y_fmt == "argmax":
            if data.y is None:
                y_final = data.target
            else:
                y_final = data.y.argmax(-1)
        else:
            y_final = data.y
        model_loss = loss_fn(modl_out, y_final)
        roar_loss = loss_fn(roar_out, y_final)

        orig_model_loss += model_loss.item()
        roar_model_loss += roar_loss.item()

    print(f"Original Model accuracy: {orig_model_correct / total} \t\t Roar Model accuracy: {roar_model_correct / total}")
    print(f"Original Model loss: {orig_model_loss / total} \t\t Roar Model loss: {roar_model_loss / total}")