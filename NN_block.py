'The block of neural networks and heads'
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class LinNN(nn.Module):
    def __init__(self, input_dim,output_dim=1):
        super(LinNN, self).__init__()
        hidden_dim=64
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.main= nn.Sequential(
                self.bn0,
                self.fc1,
                self.bn1,
                nn.ReLU()
                )
    def forward(self, x):
        x_ = self.main(x)
        x = self.fc3(x_)
        return x
class loss_head(nn.Module):
    def __init__(self,pretrained_model):
        super(loss_head, self).__init__()
        self.pretrained_model=pretrained_model.main
        in_features=pretrained_model.hidden_dim
        self.fc = nn.Sequential(
                nn.Linear(in_features, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 1)
                )
    def forward(self, x):
        x_ = self.pretrained_model(x)
        x = self.fc(x_)
        return x
class logis_head(nn.Module):
    def __init__(self,pretrained_model):
        super(logis_head, self).__init__()
        self.pretrained_model=pretrained_model.main
        in_features=pretrained_model.hidden_dim
        self.fc = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid()
                )
    def forward(self, x):
        x_ = self.pretrained_model(x)
        x = self.fc(x_)
        return x
class sel_head(nn.Module):
    def __init__(self,pretrained_model):
        super(sel_head, self).__init__()
        self.pretrained_model=pretrained_model.main
        hidden_dim=pretrained_model.hidden_dim
        output_dim=1

        #select block
        self.fc_sel = nn.Linear(hidden_dim, 16)
        self.bn_sel = nn.BatchNorm1d(16)
        self.fc_sel_2 = nn.Linear(16, 1)
        self.sel_head= nn.Sequential(
                self.fc_sel,
                self.bn_sel,
                nn.ReLU(),
                self.fc_sel_2,
                nn.Sigmoid()
                )
        #auxiliary block
        self.pred_head = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x_ = self.pretrained_model(x)
        #select
        x_sel = self.sel_head(x_)
        #auxiliary
        x_pred = self.pred_head(x_)
        return x_pred,x_sel
