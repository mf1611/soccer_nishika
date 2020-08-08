import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

class DLDataset(Dataset):
    """
    Ref: https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
    output:
        [x_num, x_cat, y]
    """

    def __init__(self, df, cont_cols, cat_cols=None, output_col=None, train_mode=True):
        self.df = df

        if output_col:
            self.y = torch.tensor(df[output_col], dtype=torch.float32)
        else:
            self.y =  torch.zeros((len(self.df), 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = cont_cols
        
        if self.cont_cols:
            self.cont_X = torch.tensor(df[self.cont_cols].astype("float").values, dtype=torch.float32)
        else:
            self.cont_X = torch.zeros((len(self.df), 1))
        
        if self.cat_cols:
          self.cat_X = torch.tensor(df[cat_cols].values, dtype=torch.long)
        else:
            self.cat_X =  torch.zeros((len(self.df), 1))

        self.train_mode = train_mode        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        if self.train_mode:
            return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]
        else:
            return [self.cont_X[idx], self.cat_X[idx]]