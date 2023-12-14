def bgnn_trainAndValidate(model, train_dl, val_dl, num_epochs, optimizer, device, loss_fn=torch.nn.functional.nll_loss, y_fmt="argmax", y_type=torch.float32):
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        for data in tqdm(train_dl, unit="batch", total=len(train_dl)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            try:
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

                try:
                    out = model(data.x, data.edge_index, data.batch)
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


            try:
                f1_y = torch.cat(val_ys, dim=0).numpy()
                f1_model = torch.cat(val_preds, dim=0).numpy()
                score_model = f1_score(f1_y, f1_model, average='micro')
            except:
                pass


            print(f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss/len(train_dl)} val loss: {val_epoch_loss/len(val_dl)}  val_acc:{val_correct/val_total}  val f1: {score_model}")
        else:
            print(
                f"Epoch {epoch}/{num_epochs} train loss: {epoch_loss / len(train_dl)}")

    return model