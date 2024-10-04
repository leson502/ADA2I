import torch
import math
import numpy as np
import random
import pickle
from tqdm import tqdm
from utils import set_seed

def load_iemocap():
    path = "data/iemocap/iemocap.pkl"
    with open(path, "rb") as f:
        unsplit = pickle.load(f)
    
    speaker_to_idx = {"M": 0, "F": 1}

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    random.shuffle(trainVid)
    testVid = list(unsplit["testVid"])

    dev_size = int(len(trainVid) * 0.1)
    
    spliter = {
        "train": trainVid[dev_size:],
        "dev": trainVid[:dev_size],
        "test": testVid
    }

    for split in data:
        for uid in tqdm(spliter[split], desc=split):
            data[split].append(
                {
                    "uid" : uid,
                    "speakers" : [speaker_to_idx[speaker] for speaker in unsplit["speaker"][uid]],
                    "labels" : unsplit["label"][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
    
    return data

def load_meld():
    path = "data/meld/meld.pkl"
    with open(path, "rb") as f:
        unsplit = pickle.load(f)

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    testVid = list(unsplit["testVid"])

    dev_size = int(len(trainVid) * 0.1)
    
    spliter = {
        "train": trainVid[dev_size:],
        "dev": trainVid[:dev_size],
        "test": testVid
    }

    for split in data:
        for uid in tqdm(spliter[split], desc=split):
            data[split].append(
                {
                    "uid" : uid,
                    "speakers" : unsplit["speakers"][uid],
                    "labels" : unsplit["label"][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
    
    return data

def load_mosei(emo="7class"):
    path = "data/mosei/mosei_data.pkl"
    with open(path, "rb") as f:
        unsplit = pickle.load(f)

    data = {
        "train": [], "dev": [], "test": [],
    }
    trainVid = list(unsplit["trainVid"])
    valVid = list(unsplit["valVid"])
    testVid = list(unsplit["testVid"])
    
    spliter = {
        "train": trainVid,
        "dev": valVid,
        "test": testVid
    }

    for split in data:
        for uid in tqdm(spliter[split], desc=split):
            data[split].append(
                {
                    "uid" : uid,
                    "speakers" : [0] * len(unsplit["speaker"][uid]),
                    "labels" : unsplit['label'][emo][uid],
                    "text": unsplit["text"][uid],
                    "audio": unsplit["audio"][uid],
                    "visual": unsplit["visual"][uid],
                    "sentence" : unsplit["sentence"][uid],
                }
            )
    
    return data

class Dataloader:
    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(data)/ self.batch_size)
        self.dataset = args.dataset
        self.embedding_dim = args.embedding_dim[self.dataset]
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor([len(s["text"]) for s in samples]).long()
        mx = torch.max(text_len_tensor).item()
        
        audio_tensor = torch.zeros((batch_size, mx, self.embedding_dim['a']))
        text_tensor = torch.zeros((batch_size, mx, self.embedding_dim['t']))
        visual_tensor = torch.zeros((batch_size, mx, self.embedding_dim['v']))

        speaker_tensor = torch.zeros((batch_size, mx)).long()

        labels = []
        utterances = []
        for i, s in enumerate(samples):
            cur_len = len(s["text"])
            utterances.append(s["sentence"])

            tmp_t = []
            tmp_a = []
            tmp_v = []
            for t, a, v in zip(s["text"], s["audio"], s["visual"]):
                tmp_t.append(torch.tensor(t))
                tmp_a.append(torch.tensor(a))
                tmp_v.append(torch.tensor(v))
                
            tmp_a = torch.stack(tmp_a)
            tmp_t = torch.stack(tmp_t)
            tmp_v = torch.stack(tmp_v)

            text_tensor[i, :cur_len, :] = tmp_t
            audio_tensor[i, :cur_len, :] = tmp_a
            visual_tensor[i, :cur_len, :] = tmp_v
            
            speaker_tensor[i, :cur_len] = torch.tensor(s["speakers"])

            labels.extend(s["labels"])

        label_tensor = torch.tensor(labels).long()
        

        data = {
            "length": text_len_tensor,
            "tensor": {
                "t": text_tensor,
                "a": audio_tensor,
                "v": visual_tensor,
            },
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "utterance_texts": utterances,
        }

        return data

    def shuffle(self):
        random.shuffle(self.data)



