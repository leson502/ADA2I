import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from .net import Net


class Ada2I(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super(Ada2I, self).__init__()

        self.net = Net(args)
        self.criterion = nn.NLLLoss()
        self.args = args
        self.modalities = args.modalities
        self.mmcosine = args.mmcosine
        self.modulation = args.modulation
        self.alpha = args.alpha


    def forward(self, data):
        out, m_out, amn_loss = self.net(data)
    
        weight_dict = self.net.get_fusion_weight()

        with torch.no_grad():
            m_out = {
                m: torch.matmul(m_out[m], torch.transpose(weight_dict[m]["weight"], 0, 1)) + weight_dict[m]["bias"]
                for m in self.modalities
            }
            prob_m = {
                m: F.log_softmax(m_out[m], dim=1)
                for m in self.modalities
            }
            scores = {
                m: sum([F.softmax(m_out[m], dim=1)[i][data["label_tensor"][i]] for i in range(out.size(0))])
                for m in self.modalities
            }

            min_score = min(scores.values())
            ratio = {
                m: scores[m] / min_score
                for m in self.modalities
            }
        
        prob = F.log_softmax(out, dim=1)

        return prob, amn_loss, prob_m, ratio
    
        
    def apply_modulation(self, ratio):

        params_dict = self.net.get_encoder_params()
        coeff = {
        m: 1 - F.tanh(self.alpha * F.relu(ratio[m], inplace=True)) if ratio[m] > 1 else 1
        for m in self.modalities
        }

        for m in self.modalities:
            for params in params_dict[m]:
                params.grad = params.grad * coeff[m]
                if self.args.normalize:
                    params.grad += torch.zeros_like(params.grad).normal_(0, params.grad.std().item() + 1e-8)
