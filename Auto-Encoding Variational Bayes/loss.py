import torch
import torch.nn.functional as F
def loss_function(recon, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD