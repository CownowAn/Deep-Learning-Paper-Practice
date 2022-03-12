import numpy as np
from prototypical_loss import prototypical_loss

def test(args, test_dataloader, model, device):
    for epoch in range(5):
        test_loss = list()
        test_acc = list()
        test_iterator = iter(test_dataloader)
        for batch in test_iterator:
            input, target = batch
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss, acc = prototypical_loss(output, target=target, n_support=args.Ns_test)
            test_loss.append(loss.item())
            test_acc.append(acc.item())
        epoch_avg_test_loss = np.mean(test_loss)
        epoch_avg_test_acc = np.mean(test_acc)
    print("Test Epoch: {} / Avg Test Loss: {} / Avg Test Accuracy: {}".format(epoch+1, epoch_avg_test_loss, epoch_avg_test_acc))