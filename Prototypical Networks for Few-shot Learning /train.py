import torch
from tqdm import tqdm
import numpy as np

def train(args, train_dataloader, model, optimizer, lr_scheduler, device):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_state = None

    best_model_path = os.path.join(args.results_root, 'best_model.pth')
    last_model_path = os.path.join(args.results_root, 'last_model.pth')

    for epoch in range(args.epochs):
        print("***** Epoch:{} *****".format(epoch+1))

        # train
        train_iterator = iter(train_dataloader)
        model.train()
        for batch in tqdm(train_iterator):
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss, acc = prototypical_loss(output, target=target, n_support=args.Ns_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        epoch_avg_train_loss = np.mean(train_loss[-args.iterations:])
        epoch_avg_train_acc = np.mean(train_acc[-args.iterations:])
        print("Epoch: {} / Avg Train Loss: {} / Avg Train Accuracy: {}".format(epoch+1, epoch_avg_train_loss, epoch_avg_train_acc))
        lr_scheduler.step()

        # validation
        val_iterator = iter(val_dataloader)
        model.eval()
        for batch in tqdm(val_iterator):
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss, acc = prototypical_loss(output, target=target, n_support=args.Ns_test)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        epoch_avg_val_loss = np.mean(val_loss[-args.iterations:])
        epoch_avg_val_acc = np.mean(val_acc[-args.iterations:])
        print("Epoch: {} / Avg Val Loss: {} / Avg Val Accuracy: {}".format(epoch+1, epoch_avg_val_loss, epoch_avg_val_acc))
        if epoch_avg_val_acc > best_acc:
            best_acc = epoch_avg_val_acc
            best_state = model.state_dict()
            torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc