import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GA(nn.Module):
    def __init__(
            self,
            seq_len,
            num_relations,
            num_heads,
            num_layers,
            device="cuda"
        ) -> None:
        super().__init__()

        self.num_heads = num_heads

        # relational embedding layer ???
        self.rel_embeddings = nn.Embedding(num_relations,
                                           num_heads,
                                           padding_idx=0)
        
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, "lin" + str(i), nn.Parameter(torch.randn((seq_len, num_heads, 1))))
            setattr(self, "bias" + str(i), nn.Parameter(torch.randn((seq_len, 1))))
        
        self.to(device)
        self.device = device

    def forward(self, x, relations):
        '''
        params:
            x:          torch.FloatTensor [B,S,D]
            relations:  torch.LongTensor  [B,S,S]
        returns:
            res:        torch.FloatTensor [B,S,D]
        '''

        # Relation-based multi head self attention

        A = self.rel_embeddings(relations) # [B, S, S, H]
        A = A.transpose(3, 2).transpose(2, 1) # [B, H, S, S]
        A = F.softmax(A, dim=-1) # row-softmax relation matrix

        for i in range(self.num_layers):
            x = self.forward_(x, A, i)

        return x
    
    def forward_(self, x, A, idx_layer):
        '''
            x:          [B,S,D]
            A:          [B,H,S,S]
            idx_layer:  number of current layer
        '''

        lin = getattr(self, "lin" + str(idx_layer))
        bias = getattr(self, "bias" + str(idx_layer))

        x = x.unsqueeze(dim=1).repeat(1, self.num_heads, 1, 1) # [B,H,S,D]
        Ax = torch.matmul(A, x) # [B,H,S,D]

        # put all the heads together through a linear transformation
        Ax = Ax.transpose(1,2).transpose(2,3) # [B,S,D,H]
        r = torch.matmul(Ax, lin).squeeze(-1) + bias # [B,S,D]
        
        return torch.tanh(r) # non-linear activation function