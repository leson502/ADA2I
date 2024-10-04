import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .utils import batch_flating

class AFW(nn.Module):
    def __init__(self, input_dim, rank, modalities, beta, nlayers, droprate=0.1) -> None:
        super(AFW, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.modalities = modalities
        self.n_modals = len(modalities)
        self.beta = beta
        self.nlayers = nlayers

        self.layers = nn.ModuleList([AFWLayer(input_dim, rank, self.n_modals, beta, droprate) 
                                    for _ in range(nlayers)])
        
    
    def forward(self, x):
        out = list(x.values())

        if self.training:
            total_loss = torch.zeros(1, device=out[0].device)
            for j in range(self.nlayers):
                out, loss = self.layers[j](out)
                total_loss += loss
        else:
            for j in range(self.nlayers):
                out = self.layers[j](out)
                
        out = {
            self.modalities[j]: out[j]
            for j in range(self.n_modals)
        }
        if self.training:
            return out, total_loss
        return out
        
        
class AFWLayer(nn.Module):
    def __init__(self, input_dim, rank, n_modals, beta, droprate=0.1) -> None:
        super(AFWLayer, self).__init__()
        self.input_dim = input_dim
        self.n_modals = n_modals

        self.attention = Attention(input_dim, rank, n_modals, beta)
        self.dropout = nn.Dropout(p=droprate)
        self.wmn = AttentionMapping(input_dim, n_modals)

    def forward(self, x):
        aware, att = self.attention(x, return_att=True)
        aware = [self.dropout(aware[j]) for j in range(self.n_modals)]

        if self.training:
            loss = self.wmn.get_loss(att, x)
            return aware, loss
        return aware



class Attention(nn.Module):
    def __init__(self, input_dim, rank, n_modals, beta) -> None:
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.n_modals = n_modals
        self.beta = beta

        self.trans_q1 = self.get_trans()
        self.trans_q2 = self.get_trans()
        self.trans_k1 = self.get_trans()
        self.trans_k2 = self.get_trans()
        self.lin_att = nn.ModuleList([Linear(rank * rank, input_dim[j]) for j in range(n_modals)])
        
        
    def get_trans(self):
        return nn.ModuleList([
            Linear(self.input_dim[j], self.rank) 
                for j in range(self.n_modals)
        ])
    
    def forward(self, x, return_att=False):
        """
        Input: List[torch.Tensor[batch_size, length, embed_dim]]
        """
        G_qk = []
        M_qk = []
        att = []
        for j in range(self.n_modals):
            G_q = self.trans_q1[j](x[j]).unsqueeze(-1) * self.trans_q2[j](x[j]).unsqueeze(-2) # mode-1 khatri-rao product
            G_k = self.trans_k1[j](x[j]).unsqueeze(-1) * self.trans_k2[j](x[j]).unsqueeze(-2)
            G_qk.append(G_q * G_k)
            M_qk.append(G_qk[j].mean(dim=1))
        
            
        for j in range(self.n_modals):
            att.append(G_qk[j])
            for l in range(self.n_modals):
                if j == l: continue
                att[j] = torch.einsum('ijkl,ilo->ijko' ,att[j], M_qk[l]) # Tensor contraction
            B, T, R1, R2 = att[j].size()
            att[j] = att[j].view(B, T, R1 * R2)
            att[j] = self.lin_att[j](att[j])
            _att = att
            att[j] = att[j] * x[j] + self.beta * x[j]

        if return_att:
            return att, _att
        return att

class AttentionMapping(nn.Module):
    def __init__(self, input_size, n_modals) -> None:
        super(AttentionMapping,self).__init__()

        self.input_size = input_size
        self.n_modals = n_modals
        
        self.fc = nn.ModuleList([nn.Linear(input_size[j], input_size[j]) for j in range(n_modals)])
        self.loss = nn.L1Loss()
    
    def get_loss(self, wi, fi):
        w_hat = [self.fc[j](fi[j]) for j in range(self.n_modals)]

        w_hat = torch.cat(w_hat, dim=1)
        wi = torch.cat(wi, dim=1)

        return self.loss(w_hat, wi)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.lin = nn.Linear(in_features, out_features, bias)
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias:
            nn.init.constant_(self.lin.bias, 0.)
    
    def forward(self, x):
        return self.lin(x)
    



