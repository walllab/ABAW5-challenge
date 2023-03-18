import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        '''
        implementation taken from --> https://github.com/kaidic/LDAM-DRW
        '''
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32, device=device)
        self.s = s
        self.weight = weight

    def forward(self, logits, labels):
        index = torch.zeros_like(logits, dtype=torch.uint8, device=device)
        index.scatter_(1, labels.data.view(-1, 1), 1)

        index_float = index.type(torch.float32)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        logits_m = logits - batch_m

        output = torch.where(index, logits_m, logits)
        return F.cross_entropy(self.s*output, labels, weight=self.weight)
    
