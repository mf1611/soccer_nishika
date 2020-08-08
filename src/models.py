import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


class DLModel(nn.Module):
    def __init__(self, num_cont_cols, num_unique_cat_cols=[], emb_size=5):
        super(DLModel, self).__init__()

        layer = []
        if len(num_unique_cat_cols)!=0:
            for num_cat in num_unique_cat_cols:
                layer.append(nn.Embedding(num_cat+1, emb_size))  # +1はテストで初めて出現するカテゴリ値用
            self.embedding = nn.ModuleList(layer)

        self.fc1 = nn.Linear(num_cont_cols, 32)
        self.bn1 = nn.BatchNorm1d(32+emb_size*len(num_unique_cat_cols))
        self.fc2 = nn.Linear(32+emb_size*len(num_unique_cat_cols), 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        

    def forward(self, x_num, x_cat=None):
        x = F.relu(self.fc1(x_num))

        for i in range(x_cat.shape[1]):
            x = torch.cat([x, self.embedding[i](x_cat[:,i]).squeeze()], dim=-1)

        x = F.dropout(x, p=0.3, training=self.training)
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.bn2(x)
        x = self.fc3(x)
        return x.squeeze()