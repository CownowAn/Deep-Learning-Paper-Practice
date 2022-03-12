import easydict
import torch
import numpy as np

# set arguments
args = easydict.EasyDict({
    'root': 'data', # path to dataset
    'results_root': 'results', # root to store models, loss and accuracies
    'epochs': 20,
    'learning_rate': 0.001,
    'lr_scheduler_step': 20,
    'lr_scheduler_gamma': 0.5,
    'iterations': 100,
    'Nc_train': 60,
    'Ns_train': 5,
    'Nq_train': 5,
    'Nc_test': 5,
    'Ns_test': 5,
    'Nq_test': 15,
    'manual_seed': 423,
    'use_cuda': True
})

device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

# init seed
torch.cuda.cudnn_enabled = False

seed = args.manual_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)