# -*- coding: UTF-8 -*-   # Prevent garbled text when inputting Chinese

from __future__ import division  # Declare to eliminate ambiguity in division

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
# from matplotlib.mlab import griddata
from matplotlib.path import Path

##from scipy.interpolate import griddata
from scipy import interpolate
from scipy.interpolate import Rbf
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.optim
from torch.nn import init
from torch_geometric.data import Data

# Read the first Excel table (meteorological features and target column)
data = pd.read_excel(
    '/Users/zhangjingshuai/Library/Application Support/JetBrains/PyCharmCE2024.1/scratches/finaldata.xlsx', header=0)

# Read the second Excel table (containing connection information, each row represents a connection)
graph_data = pd.read_csv(
    '/Users/zhangjingshuai/Library/Application Support/JetBrains/PyCharmCE2024.1/scratches/sorted_full_edge_index.csv',
    header=0)
# Assume that the two Excel tables have been merged based on node IDs and contain connection information.
# Now we need to organize the data into the Data object required by PyTorch Geometric.

# Separate features and target column
x = data.iloc[:, 0:9].values
y = data.iloc[:, 12].values

# Data normalization
# Normalize the feature x
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
x = (x - mean_x) / std_x

# Create a PyTorch Geometric data object for the connection information
edge_index = graph_data[['start_node', 'end_node']].values.T  # Transpose to match the requirements of PyTorch Geometric
edge_index = torch.tensor(edge_index, dtype=torch.long)

x = torch.tensor(np.array(x), dtype=torch.float)  # Node feature matrix [N, D]
edge_index = torch.tensor(edge_index)  # Edge Index [2, E]
y = torch.tensor(y, dtype=torch.float).T  # Ensure data.y shape is [N]
data = Data(x=x, edge_index=edge_index, y=y)

train_ratio = 0.6  # Training set ratio
val_ratio = 0.2  # Validation set ratio
test_ratio = 0.2  # Test set ratio

N = data.x.shape[0]
num_train = int(train_ratio * N)
num_val = int(val_ratio * N)
num_test = int(test_ratio * N)

train_index = torch.randperm(N)[:num_train]
val_index = torch.randperm(N)[:num_val]
test_index = torch.randperm(N)[:num_test]

# #================================================================================

hidden_channels_N = 128


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = torch.mm(x, self.weight)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    def __init__(self, data, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #         self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #         self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #         self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #         x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #         x = F.dropout(x, p=0.5, training=self.training)

        #         x = self.conv3(x, edge_index)
        #         x = F.relu(x)
        # #         x = F.dropout(x, p=0.5, training=self.training)

        #         x = self.conv4(x, edge_index)
        #         x = F.relu(x)
        # #         x = F.dropout(x, p=0.1, training=self.training)

        #         x = self.conv5(x, edge_index)
        #         x = F.relu(x)
        # #         x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv6(x, edge_index)
        #         return F.log_softmax(x, dim=1)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data, hidden_channels=hidden_channels_N).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = out.squeeze()
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])  # Change nll_loss to mse_loss
    loss = F.mse_loss(out[~torch.isnan(data.y)], data.y[~torch.isnan(data.y)])
    loss.backward()
    optimizer.step()
    return loss


print('Training data based on the Traditional GCN model')

# Initialize the highest accuracy before the training loop starts
best_acc = 0.0

for epoch in range(1, 50000):
    loss = train(data)
    if epoch % 500 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

print('OK')

# Use the trained model for prediction
# model.eval()
# with torch.no_grad():
#     pred = model(data.x, data.edge_index)
#     pred = pred.flatten()
#     print('Predicted values:', pred)
