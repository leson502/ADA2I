import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import batch_flating
from .encoder import Transfomer_Encoder, LSTM_Encoder
from .AFW import AFW

class Net(nn.Module):
    def __init__(self, args) -> None:
        super(Net, self).__init__()

        if args.emotion == "7class":
            self.label_dict = args.dataset_label_dict["mosei7"]
        elif args.emotion:
            self.label_dict = args.dataset_label_dict["mosei2"]
        else:
            self.label_dict = args.dataset_label_dict[args.dataset]

        tag_size = len(self.label_dict)
        self.num_modal = len(args.modalities)
        self.args = args
        self.modalities = args.modalities
        self.embedding_dim = args.embedding_dim[args.dataset]
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.dropout = args.drop_rate
        
        self.encoder = nn.ModuleDict()

        if not args.no_mmt:
            print("Net -> Using AFW")
            self.mmt = AFW([self.hidden_dim] * self.num_modal, args.tensor_rank, self.modalities, args.beta, args.mmt_nlayers, self.dropout)
        else:
            self.mmt = None

        self.fc_out = nn.Linear(self.hidden_dim * self.num_modal, tag_size)

        for m in self.modalities:
            self.encoder.add_module(m, self.get_encoder(m))
        
        if self.args.mmcosine:
            print("Net --> Using AMW")
    
    def get_encoder(self, m):
        print(f"Net --> {m} encoder: {self.args.encoder_modules}")
        if self.args.encoder_modules == "transformer":
            return Transfomer_Encoder(
                self.embedding_dim[m], 
                self.hidden_dim,
                self.args.encoder_nlayers,
                dropout=self.dropout )
        
        elif self.args.encoder_modules == "lstm":
            return LSTM_Encoder(
                self.embedding_dim[m], 
                self.hidden_dim,
                self.args.encoder_nlayers,
                dropout=self.dropout )



    def forward(self, data):
        x = data["tensor"]
        lengths = data["length"]

        encoded = {}

        for m in self.modalities:
            encoded[m] = self.encoder[m](x[m], lengths)

        amn_loss = 0
        if self.mmt:
            if self.training:
                out, amn_loss = self.mmt(encoded)
            else:
                out = self.mmt(encoded)
        else:
            out = encoded
        
        out = batch_flating(out, lengths, self.modalities, self.device)
        _out = out

        if self.args.mmcosine:
            weight_dict = self.get_fusion_weight()
            out = {
                m: torch.matmul(F.normalize(out[m], dim=1), F.normalize(torch.transpose(weight_dict[m]["weight"], 0, 1), dim=1)) + weight_dict[m]["bias"]
                for m in self.modalities
            }
            stack = torch.stack(list(out.values()), dim=1)
            out = torch.sum(stack, dim=1)
        else:
            out = torch.cat(list(out.values()), dim=-1)
            out = self.fc_out(out)

        return out, _out, amn_loss
    
    def get_encoder_params(self):
        params_dict = {
            m:[] for m in self.modalities
        }

        for m in self.modalities:
            for params in self.encoder[m].parameters():
                params_dict[m].append(params)
        return params_dict
    
    def get_fusion_weight(self):
        weight_dict = {
            m: {} for m in self.modalities
        }
        
        for j, m in enumerate(self.modalities):
            weight_dict[m]["weight"] = self.fc_out.weight[:, j * self.hidden_dim: (j + 1) * self.hidden_dim]
            weight_dict[m]["bias"] = self.fc_out.bias / self.num_modal
    
        return weight_dict