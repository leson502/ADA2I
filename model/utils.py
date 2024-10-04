import torch
import numpy as np

from torch import tensor
def batch_flating(batches, lengths, modalities, device):
     
     features = {
          m:[] for m in modalities
     }

     lengths = lengths.tolist()
     batch_size = len(lengths)

     for j in range(batch_size):
          cur_len = lengths[j]
          for m in modalities:
               features[m].append(batches[m][j,:cur_len])

     for m in modalities:
          features[m] = torch.cat(features[m], dim=0).to(device)

     return features




if __name__ == '__main__':
    pass