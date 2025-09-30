
# vfl_train_test.py
import argparse
import csv
import os
import time
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

from config import DEFAULTS, DATASETS, MODELS
from models_snn import (
    Client,
    ServerNoSplit,
    VGGClientSNN,
    SResnetClient,
)

from sklearn.metrics import precision_score, recall_score, confusion_matrix

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# --------------- helpers ---------------
class ShuffledDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


def split_data(dataset, worker_list=None, clients=2):
    """
    Vertical split along image width among clients.
    Returns dict of lists of tensors for each worker and label/index lists (unused downstream).
    """
    if worker_list is None:
        worker_list = list(range(0, clients))

    idx = 0
    dic_single_datasets = {w: [] for w in worker_list}
    label_list = []
    index_list = []
    index_list_UUID = []
    for tensor, label in dataset:
        height = tensor.shape[-1] // len(worker_list)
        i = 0
        uuid_idx = uuid4()
        for worker in worker_list[:-1]:
            dic_single_datasets[worker].append(tensor[:, :, height * i: height * (i + 1)])
            i += 1
        dic_single_datasets[worker_list[-1]].append(tensor[:, :, height * i:])
        label_list.append(torch.Tensor([label]))
        index_list_UUID.append(uuid_idx)
        index_list.append(torch.Tensor([idx]))
        idx += 1

    return dic_single_datasets, label_list, index_list, index_list_UUID


# --------------- train/test ---------------
def train_one_epoch(epoch, client_instances, clients, client_optimizers,
                    server, data_loaders, label_loader, batch_size):
    start_time = time.time()
    client_times = []

    dataset_length = len(data_loaders[0].dataset)
    indices = torch.randperm(dataset_length).tolist()
    shuffled_data_datasets = [ShuffledDataset(dl.dataset, indices) for dl in data_loaders]
    shuffled_label_dataset = ShuffledDataset(label_loader.dataset, indices)

    shuffled_data_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for dataset in shuffled_data_datasets
    ]
    shuffled_label_loader = DataLoader(shuffled_label_dataset, batch_size=batch_size,
                                       shuffle=False, drop_last=True)

    for client in client_instances:
        client.model.train()

    losses = []
    correct = 0
    total = 0

    for data_tuple in zip(*shuffled_data_loaders, shuffled_label_loader):
        batch_data = data_tuple[:-1]
        labels = data_tuple[-1].to(DEVICE)

        outputs_list = []
        for i, client in enumerate(client_instances):
            t0 = time.time()
            out = client.forward(batch_data[i])
            client_times.append(time.time() - t0)
            outputs_list.append(out.to(DEVICE))

        detached = [o.detach().requires_grad_(True) for o in outputs_list]
        loss, sum_logits, client_grads = server.update(clients, detached, labels, True)

        for opt in client_optimizers:
            opt.zero_grad()

        for o, g in zip(outputs_list, client_grads):
            o.backward(g)

        for opt in client_optimizers:
            opt.step()

        losses.append(loss.item())

        _, predicted = torch.max(sum_logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100.0 * correct / total
    total_time = time.time() - start_time
    avg_client_time = sum(client_times) / len(client_times) if client_times else 0.0
    print(f"Epoch {epoch + 1}, Train Acc: {train_acc:.2f}%")

    return losses, float(np.mean(losses)), train_acc, avg_client_time, total_time


@torch.no_grad()
def evaluate(clients, server, client_instances, val_loaders, t_local):
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_data, labels in zip(zip(*[dl for dl in val_loaders]), t_local):
        labels = labels.to(DEVICE)
        client_outputs = []
        for client, client_data in zip(client_instances, batch_data):
            client_outputs.append(client.forward(client_data).to(DEVICE))

        loss, output, _ = server.update(clients, client_outputs, labels, False)
        test_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

        all_preds.extend(pred.view(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_loss /= len(t_local.dataset)
    accuracy = 100.0 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return test_loss, accuracy, precision, recall, conf_matrix, f1_score


# --------------- main ---------------
def main(args):
    # Resolve config
    ds_key = args.dataset.lower()
    if ds_key not in DATASETS:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Choose from: {list(DATASETS.keys())}")

    model_key = args.model.lower()
    if model_key not in MODELS:
        raise ValueError(f"Unsupported model '{args.model}'. Choose from: {list(MODELS.keys())}")

    ds_cfg = DATASETS[ds_key]
    mdl_cfg = MODELS[model_key]

    num_classes = ds_cfg["num_classes"]
    transform = ds_cfg["transform"]

    # Hyperparams (dataset defaults -> global defaults), overridden by CLI if provided
    batch_size = args.batch_size or DEFAULTS["batch_size"]
    epochs = args.epochs or DEFAULTS["epochs"]
    img_size = DEFAULTS["img_size"]
    leak_mem = args.leak_mem if args.leak_mem is not None else ds_cfg.get("leak_mem", DEFAULTS["leak_mem"])
    lr = args.lr if args.lr is not None else ds_cfg.get("learning_rate", DEFAULTS["learning_rate"])
    momentum = args.momentum if args.momentum is not None else DEFAULTS["momentum"]
    weight_decay = args.weight_decay if args.weight_decay is not None else DEFAULTS["weight_decay"]
    timesteps = args.timesteps if args.timesteps is not None else ds_cfg.get("timesteps", DEFAULTS["timesteps"])

    # Data
    if ds_key == "cifar10":
        trainset = torchvision.datasets.CIFAR10("CIFAR10", download=True, train=True, transform=transform)
        valset = torchvision.datasets.CIFAR10("CIFAR10", download=True, train=False, transform=transform)
        labels = torchvision.datasets.CIFAR10("CIFAR10", download=True, train=True).targets[:50_000]
        labels_val = torchvision.datasets.CIFAR10("CIFAR10", download=True, train=False).targets[:10_000]
    else:
        trainset = torchvision.datasets.CIFAR100("CIFAR100", download=True, train=True, transform=transform)
        valset = torchvision.datasets.CIFAR100("CIFAR100", download=True, train=False, transform=transform)
        labels = torchvision.datasets.CIFAR100("CIFAR100", download=True, train=True).targets[:50_000]
        labels_val = torchvision.datasets.CIFAR100("CIFAR100", download=True, train=False).targets[:10_000]

    # Vertical split among workers
    clients = args.clients
    img_splits, _, _, _ = split_data(trainset, clients=clients)
    img_splits_val, _, _, _ = split_data(valset, clients=clients)

    data_loaders = [DataLoader(img_splits[w], batch_size=batch_size, drop_last=True, shuffle=False)
                    for w in range(clients)]
    val_loaders = [DataLoader(img_splits_val[w], batch_size=batch_size, drop_last=True, shuffle=False)
                   for w in range(clients)]

    # Just label tensors for loaders
    label_loader = DataLoader(labels, batch_size=batch_size, drop_last=True, shuffle=False)
    t_local = DataLoader(labels_val, batch_size=batch_size, drop_last=True, shuffle=False)

    # Compute Y_img_size for each worker slice
    base_size = img_size // clients
    sizes = [base_size] * clients
    sizes[-1] += (img_size % clients)

    # Build clients
    client_models = []
    for i in range(clients):
        if model_key == "vggsnn":
            model = VGGClientSNN(
                timesteps=timesteps,
                leak_mem=leak_mem,
                X_img_size=img_size,
                Y_img_size=sizes[i],
                num_cls=num_classes,
                fc_dim=MODELS["vggsnn"]["fc_dim"],
                dropout=DEFAULTS["dropout"],
            )
        else:  # sresnet
            model = SResnetClient(
                n=MODELS["sresnet"]["n"],
                nFilters=MODELS["sresnet"]["nFilters"],
                num_steps=timesteps,
                leak_mem=leak_mem,
                img_size=img_size,
                Y_img_size=sizes[i],
                num_cls=num_classes,
                boosting=False,
                use_poisson=True,
            )
        client_models.append(model)

    # Optimizers and client wrappers
    client_optimizers = [optim.SGD(m.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                         for m in client_models]
    client_instances = [Client(f"client{i}", data_loaders[i], client_models[i]) for i in range(clients)]

    # No-split server
    server = ServerNoSplit(nn.CrossEntropyLoss())

    # Train/eval
    best_acc = 0.0
    best_epoch_metrics = {}
    avg_client_times = []
    total_times = []

    for epoch in range(epochs):
        tr_losses, tr_mean, tr_acc, avg_ct, total_t = train_one_epoch(
            epoch, client_instances, clients, client_optimizers, server, data_loaders, label_loader, batch_size
        )
        avg_client_times.append(avg_ct)
        total_times.append(total_t)

        te_loss, te_acc, te_prec, te_rec, te_cm, te_f1 = evaluate(
            clients, server, client_instances, val_loaders, t_local
        )

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch_metrics = {
                "epoch": epoch + 1,
                "accuracy": te_acc,
                "precision": te_prec,
                "recall": te_rec,
                "f1_score": te_f1,
                "conf_matrix": te_cm,
            }

        print(f"[Val] Epoch {epoch + 1} | Acc: {te_acc:.2f}% | Prec: {te_prec:.3f} | Rec: {te_rec:.3f} | F1: {te_f1:.3f}")

    # Save results
    out_csv = f"perf_nosplit_{ds_key}_{model_key}.csv"
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Metric", "Value"])
        writer.writerow(["Dataset", ds_key])
        writer.writerow(["Model", model_key])
        writer.writerow(["clients", clients])
        writer.writerow(["epochs", epochs])
        writer.writerow(["batch_size", batch_size])
        writer.writerow(["timesteps", timesteps])
        writer.writerow(["lr", lr])
        writer.writerow(["momentum", momentum])
        writer.writerow(["weight_decay", weight_decay])
        writer.writerow(["leak_mem", leak_mem])

        writer.writerow(["Best Accuracy", best_acc])
        if best_epoch_metrics:
            writer.writerow(["Best Epoch", best_epoch_metrics["epoch"]])
            writer.writerow(["Precision", best_epoch_metrics["precision"]])
            writer.writerow(["Recall", best_epoch_metrics["recall"]])
            writer.writerow(["F1", best_epoch_metrics["f1_score"]])
            cm = best_epoch_metrics["conf_matrix"]
            cm_str = '\n'.join(['\t'.join(map(str, row)) for row in cm])
            writer.writerow(["Confusion Matrix", cm_str])
        writer.writerow(["Average Client Time (s)", np.mean(avg_client_times) if avg_client_times else 0.0])
        writer.writerow(["Total Training Time (s)", np.sum(total_times) if total_times else 0.0])

    print(f"Saved metrics to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-split VFL SNN (CIFAR-10/100)")

    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"],
                        help="Dataset to use.")
    parser.add_argument("--model", type=str, default="vggsnn", choices=["vggsnn", "sresnet"],
                        help="Client model type.")
    parser.add_argument("--clients", type=int, default=2, help="Number of vertical clients.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override time steps.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--momentum", type=float, default=None, help="Override momentum.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay.")
    parser.add_argument("--leak_mem", type=float, default=None, help="Override membrane leak.")

    args = parser.parse_args()
    main(args)

