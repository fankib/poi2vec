import torch

# Parameters
# ==================================================
#ltype = torch.cuda.LongTensor
#ftype = torch.cuda.FloatTensor
#ltype = torch.LongTensor
#ftype = torch.FloatTensor
ltype = None # set in training
ftype = None

dataset = "../dataset/loc-gowalla_totalCheckins.txt"

# Model Hyperparameters
feat_dim = 200
route_depth = 16
route_count = 4
context_len = 32

# Weight init
weight_m = 0
weight_v = 0.1

# Training Parameters
batch_size = 128
num_epochs = 30
learning_rate = 1e3
momentum = 0.8
evaluate_every = 3
