from comet_ml import Experiment
import numpy as np
import argparse
import os
import time
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from utils import set_seed
from dataloader import Dataloader, load_iemocap, load_meld, load_mosei
from model import Ada2I
from optimizer import Optimizer
from sklearn import metrics


def train(model: nn.Module, 
          train_set: Dataloader, 
          dev_set: Dataloader, 
          test_set: Dataloader, 
          criterion: nn.Module, 
          optimizer: Optimizer,
          logger: Experiment, 
          args):
    
    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    tanh = nn.Tanh()
    relu = nn.ReLU(inplace=True)

    modalities = args.modalities
    device = args.device
    dev_f1, loss = [], []
    best_dev_f1 = None
    best_test_f1 = None
    best_state = None
    best_epoch = None

    optimizer.set_parameters(model.parameters(), args.optimizer)
    scheduler = optimizer.get_scheduler(args.scheduler)

    early_stopping_count = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss = 0
        m_loss = {m: 0 for m in modalities}
        model.train()
        train_set.shuffle()
        for idx in tqdm(range(len(train_set)), desc=f"train_epoch {epoch}"):
            model.zero_grad()

            data = train_set[idx]
            for k, v in data.items():
                if k == "utterance_texts": continue
                if k == "tensor":
                    for tk, tv in data[k].items():
                        data[k][tk] = tv.to(device)
                else:
                    data[k] = v.to(device)

            prob, amn_loss, m_prob, ratio = model(data)

            nll = criterion(prob, data["label_tensor"]) + amn_loss
            for m in modalities:
                m_loss[m] += criterion(m_prob[m], data["label_tensor"]).item()
            loss += nll.item()
            nll.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm_max, norm_type=args.grad_norm)

            if args.modulation and args.start_mod <= epoch <= args.end_mod:
                model.apply_modulation(ratio)

            optimizer.step()

            del data

        end_time = time.time()
        print(f"[Epoch {epoch}] [Train Loss: {loss}] [Time: {end_time - start_time}]")
        print(f"[Ratio {ratio}]")
        print(f"[Unimodal loss: {m_loss}]")

        dev_f1, dev_loss = evalute(model, dev_set, criterion, args, logger, test=False)
        print(f"[Dev Loss: {dev_loss}]\n[Dev F1: {dev_f1}]")
        scheduler.step(dev_loss)

        if args.comet:
            logger.log_metric("train_loss", loss, epoch=epoch)
            logger.log_metric("dev_loss", dev_loss, epoch=epoch)
            logger.log_metric("dev_f1", dev_f1, epoch=epoch)
            for m in modalities:
                logger.log_metric(f"loss_{m}", m_loss[m], epoch=epoch)
                logger.log_metric(f"ratio_{m}", ratio[m], epoch=epoch)

        if best_dev_f1 is None or dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1, _ = evalute(model, test_set, criterion, args, logger, test=False)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            early_stopping_count = 0
        else:
            early_stopping_count += 1
        
        if early_stopping_count == args.early_stopping:
            print(f"Early stopping at epoch: {epoch}")
            break
    
    # Best state
    print(f"Best model at epoch: {best_epoch}")
    print(f"Best dev F1: {best_dev_f1}")
    model.load_state_dict(best_state)
    f1, _ = evalute(model, test_set, criterion, args, logger, test=True)
    print(f"Best test F1: {f1}")

    if args.comet:
        logger.log_metric("best_test_f1", f1, epoch=epoch)
        logger.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
        
    return best_dev_f1, best_test_f1, best_state


def evalute(model, testset, criterion, args, logger:Experiment, test=True):

    log_softmax = nn.LogSoftmax(dim=1)
    device = args.device
    model.eval()

    if args.emotion == "7class":
        label_dict = args.dataset_label_dict["mosei7"]
    elif args.emotion:
        label_dict = args.dataset_label_dict["mosei2"]
    else:
        label_dict = args.dataset_label_dict[args.dataset]
    
    labels_name = list(label_dict.keys())

    with torch.no_grad():
        golds, preds = [], []
        loss = 0
        for idx in range(len(testset)):
            data = testset[idx]
            for k, v in data.items():
                if k == "utterance_texts": continue
                if k == "tensor":
                    for tk, tv in data[k].items():
                        data[k][tk] = tv.to(device)
                else:
                    data[k] = v.to(device)
            
            golds.append(data["label_tensor"].to("cpu"))
            
            prob, _, _, _ = model(data)
            y_hat = torch.argmax(prob, dim=-1)
            preds.append(y_hat.detach().to("cpu"))

            nll = criterion(prob, data["label_tensor"])
            loss += nll.item()

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()

        f1 = metrics.f1_score(golds, preds, average="weighted")

        if test:
            print(metrics.classification_report(golds, preds, target_names=labels_name, digits=4))
            if args.comet:
                logger.log_confusion_matrix(golds.tolist(), preds, labels=list(labels_name), overwrite=True)

        return f1, loss
    
def get_argurment():
    parser = argparse.ArgumentParser()

    #________________________________ Trainning Setting ____________________________________
    parser.add_argument(
        "--project", type=str, default="erc-gm"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iemocap", "iemocap_4", "meld", "mosei"],
        default="iemocap",
    )

    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--devset_ratio", type=float, default=0.1
    )

    parser.add_argument(
        "--modalities",
        type=str,
        choices=["atv", "at", "av", "tv", "a", "t", "v"],
        default="atv",
    )

    parser.add_argument(
        "--data_dir_path", type=str, default="data",
    )

    parser.add_argument(
        "--seed", default=12,
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam", "adamw", "rmsprop"],
        default="adam",
    )

    parser.add_argument(
        "--scheduler", type=str, choices="reduceLR", default="reduceLR",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.0002,
    )

    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
    )

    parser.add_argument(
        "--early_stopping", type=int, default=20,
    )

    parser.add_argument(
        "--batch_size", type=int, default=16,
    )

    parser.add_argument(
        "--epochs", type=int, default=50,
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"]
    )

    parser.add_argument(
        "--modulation", action="store_true", default=False
    )

    parser.add_argument(
        "--alpha", type=float, default=0.5
    )

    parser.add_argument(
        "--start_mod", type=int, default=1
    )

    parser.add_argument(
        "--end_mod", type=int, default=3000
    )

    parser.add_argument(
        "--normalize", action="store_true", default=False
    )

    parser.add_argument(
        "--comet", action="store_true", default=False
    )

    parser.add_argument(
        "--mmcosine", action="store_true", default=False
    )

    parser.add_argument(
        "--grad_clipping", action="store_true", default=False,
    )

    parser.add_argument(
        "--grad_norm", type=float, default=2.0,
    )

    parser.add_argument(
        "--grad_norm_max", type=float, default=2.0,
    )

    #________________________________ Model Setting ____________________________________
    
    parser.add_argument(
        "--encoder_modules", type=str, default="transformer", choices=["transformer", "lstm"]
    )

    parser.add_argument(
        "--encoder_nlayers", type=int, default=2,
    )

    parser.add_argument(
        "--mmt_nlayers", type=int, default=2,
    )

    parser.add_argument(
        "--tensor_rank", type=int, default=8,
    )

    parser.add_argument(
        "--beta", type=float, default=0.7,
    )

    parser.add_argument(
        "--nheads", type=int, default=2
    )

    parser.add_argument(
        "--hidden_dim", type=int, default=200,
    )

    parser.add_argument(
        "--fusion", type=str, default="concat", choices=["sum", "concat", "film", "gated"]
    )

    parser.add_argument(
        "--drop_rate", type=float, default=0.5,
    )

    parser.add_argument(
        "--no_mmt", action="store_true", default=False,
    )

    args, unknown = parser.parse_known_args()

    args.embedding_dim = {
        "iemocap": {
            "a": 512,
            "t": 768,
            "v": 1024,
        },
        "mosei": {
            "a": 512,
            "t": 768,
            "v": 1024,
        },
        "meld": {
            "a": 300,
            "t": 768,
            "v": 342,
        }
    }

    args.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "meld": {"neu": 0, "sup": 1 , "fea": 2, "sad": 3, "joy": 4, "dis": 5, "ang": 6},
            "mosei7": {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,},
            "mosei2": {
                "Negative": 0,
                "Positive": 1,},
    }

    args.dataset_num_speakers = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei7":1,
            "mosei2": 1,
        }
    
    if args.seed == "time":
        args.seed = int(datetime.now().timestamp())
    else: args.seed = int(args.seed)
    
    return args


def main(args):
    set_seed(args.seed)
    
    if args.dataset == "iemocap":
        data = load_iemocap()
    if args.dataset == "meld":
        data = load_meld()
    if args.dataset == "mosei":
        data = load_mosei(emo=args.emotion)

    train_set = Dataloader(data["train"], args)
    dev_set = Dataloader(data["dev"], args)
    test_set = Dataloader(data["test"], args)

    optim = Optimizer(args.learning_rate, args.weight_decay)
    model = Ada2I(args).to(args.device)

    if args.comet:
        logger = Experiment(project_name=args.project,
                            auto_param_logging=False,
                            auto_metric_logging=False)
        logger.log_parameters(args)
    else:
        logger = None

    dev_f1, test_f1, state = train(model, train_set, dev_set, test_set, nn.NLLLoss(), optim, logger, args)

    checkpoint_path = os.path.join("checkpoint", f"{args.dataset}_best_f1.pt")
    torch.save({"args": args, "state_dict": state}, checkpoint_path)


if __name__ == "__main__":
    args = get_argurment()
    main(args)

    



