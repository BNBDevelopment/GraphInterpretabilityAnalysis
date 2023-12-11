import os.path

import torch
from tqdm import tqdm


def train(model, configuration):
    n_epochs = configuration['n_epochs']
    optimizer = configuration['optimizer']
    #data = configuration['data']

    loss_fn = configuration['loss_fn']

    train_dl = configuration['train_data']
    val_dl = configuration['val_data']

    max_trn_acc = 0
    max_val_acc = 0
    old_save_path = ""

    for epoch in range(1, n_epochs+1):
        model.train()
        #for data in train_dl:
        train_loss = 0
        train_count_correct = 0
        for data in tqdm(train_dl, unit="batch", total=len(train_dl)):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, data.edge_attr)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()

            train_count_correct += torch.sum(torch.argmax(out, dim=1) == data.y).item()
            train_loss += loss.item()


        model.eval()
        val_loss = 0
        val_count_correct = 0
        with torch.no_grad():
            for data in tqdm(val_dl, unit="batch", total=len(val_dl)):
                out = model(data.x, data.edge_index, data.batch, data.edge_attr)
                loss = loss_fn(out, data.y)

                val_count_correct += torch.sum(torch.argmax(out, dim=1) == data.y).item()
                val_loss += loss.item()
        print(f"Epoch {epoch} \n"
              f"\t average training loss: {train_loss/len(train_dl)}"
              f"\t average training accuracy: {train_count_correct/len(train_dl.dataset)}"
              f"\t average validation loss: {val_loss/len(val_dl)}"
              f"\t average validation accuracy: {val_count_correct/len(val_dl.dataset)}")


        val_acc = val_count_correct/len(val_dl.dataset)
        model_name = f"{configuration['model_name']}_{'dataset_name'}_Epoch_{epoch}_ValAcc_{val_acc}.pt"
        model_save_path = configuration['model_save_path']

        if configuration['save_strategy'] == "max_val_acc":
            if val_acc > max_val_acc:
                print(f"Saving at epoch {epoch} with validation accuracy {val_acc} better than old validation accuracy {max_val_acc:.4f}")
                if os.path.exists(old_save_path):
                    os.remove(old_save_path)
                max_val_acc = val_acc
                old_save_path = model_save_path + model_name
                torch.save(model, model_save_path + model_name)


    return train_loss, train_count_correct, val_loss, val_count_correct