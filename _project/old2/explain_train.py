import os.path

import torch
from tqdm import tqdm


def train(model, configuration):
    model.train()
    n_epochs = configuration['n_epochs']
    optimizer = configuration['optimizer']
    #data = configuration['data']

    loss_fn = configuration['loss_fn']

    train_dl = configuration['train_data']
    val_dl = configuration['val_data']

    max_trn_acc = 0
    if configuration['save_strategy'] == "max_val_acc":
        val_criteria = 0
    else:
        val_criteria = 99999999
    old_save_path = ""

    for epoch in range(1, n_epochs+1):
        model.train()
        #for data in train_dl:
        train_loss = 0
        train_count_correct = 0
        total_count = 0

        for data in tqdm(train_dl, unit="batch", total=len(train_dl)):
            optimizer.zero_grad()
            data = data.to(configuration['device'])
            out = model(data.x.to(torch.float32), data.edge_index, data.batch, data.edge_attr)
            loss = loss_fn(out, data.y.argmax(-1))
            loss.backward()
            optimizer.step()

            if len(data.y.shape) > 1:
                train_count_correct += torch.sum(torch.argmax(out, dim=1) == torch.argmax(data.y, dim=1)).item()
            else:
                train_count_correct += torch.sum(torch.argmax(out, dim=1) == data.y).item()
            train_loss += loss.item()
            total_count += data.y.shape[0]


        model.eval()
        val_loss = 0
        val_count_correct = 0
        val_count = 0
        with torch.no_grad():
            for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
                data = data.to(configuration['device'])
                out = model(data.x.to(torch.float32), data.edge_index, data.batch, data.edge_attr)
                loss = loss_fn(out, data.y.argmax(-1))

                if len(data.y.shape) > 1:
                    val_count_correct += torch.sum(torch.argmax(out, dim=1) == torch.argmax(data.y, dim=1)).item()
                else:
                    val_count_correct += torch.sum(torch.argmax(out, dim=1) == data.y).item()
                val_loss += loss.item()
                val_count += data.y.shape[0]

        val_acc = val_count_correct / val_count
        print(f"Epoch {epoch} \n"
              f"\t average training loss: {train_loss/len(train_dl)}"
              f"\t average training accuracy: {train_count_correct/total_count}"
              f"\t average validation loss: {val_loss/len(val_dl)}"
              f"\t average validation accuracy: {val_acc}")



        avg_val_loss = val_loss / len(val_dl)

        model_save_path = configuration['model_save_path']

        if configuration['save_strategy'] == "max_val_acc":
            model_name = f"{configuration['model_name']}_{configuration['dataset_name']}_Epoch_{epoch}_ValAcc_{val_acc}.pt"
            if val_acc > val_criteria:
                print(f"Saving at epoch {epoch} with validation accuracy {val_acc} better than old validation accuracy {val_criteria:.4f}")
                if os.path.exists(old_save_path):
                    os.remove(old_save_path)
                val_criteria = val_acc
                old_save_path = model_save_path + model_name
                torch.save(model, model_save_path + model_name)
        elif configuration['save_strategy'] == "min_val_loss":
            model_name = f"{configuration['model_name']}_{configuration['dataset_name']}_Epoch_{epoch}_ValLoss_{avg_val_loss}.pt"
            if avg_val_loss < val_criteria:
                print(f"Saving at epoch {epoch} with validation loss {avg_val_loss} better than old validation loss {val_criteria:.4f}")
                if os.path.exists(old_save_path):
                    os.remove(old_save_path)
                val_criteria = avg_val_loss
                old_save_path = model_save_path + model_name
                torch.save(model, model_save_path + model_name)


    return train_loss, train_count_correct, val_loss, val_count_correct

def test(model, configuration):
    test_data = configuration['test_data']
    device = configuration['device']

    model.eval()
    total = 0

    test_mse = 0
    correct = 0

    for test_batch in test_data:
        test_batch = test_batch.to(device)
        pred = model(x=test_batch.x.to(torch.float32), edge_index=test_batch.edge_index, batch_mapping=test_batch.batch)

        total += test_batch.y.shape[0]
        if configuration['n_classes'] == 1:
            test_mse += sum(pred).item()
        else:
            pred = pred.argmax(-1)

            if len(test_batch.y.shape) > 1:
                correct += (pred == torch.argmax(test_batch.y, dim=1)).sum().item()
            else:
                correct += (pred == test_batch.y).sum().item()


    if configuration['n_classes'] == 1:
        mse = test_mse / total
        print(f'Full Model Test MSE: {mse:.4f}')
    else:
        acc = correct / total
        print(f'Full Model Test Accuracy: {acc:.4f}')