
# vfl_train_test.py
import argparse
import csv
import time
from uuid import uuid4
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import torchvision
import torchvision.transforms as transforms

from config import DEVICE, BATCH_SIZE, EPOCHS, IMG_SIZE, DATA_ROOT, CFG, DEFAULT_LEAK_MEM
from models_snn import SResnet, ServerM, VGGSNNClient, VGGSNNServer



def build_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def load_dataset(name: str, root: str, transform: transforms.Compose):
    name = name.lower()
    if name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root, download=True, train=True, transform=transform)
        testset  = torchvision.datasets.CIFAR10(root, download=True, train=False, transform=transform)
        return trainset, testset, 10, f"vfl_results_{name}.csv"
    elif name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root, download=True, train=True, transform=transform)
        testset  = torchvision.datasets.CIFAR100(root, download=True, train=False, transform=transform)
        return trainset, testset, 100, f"vfl_results_{name}.csv"
    else:
        raise ValueError("Unsupported dataset. Use 'cifar10' or 'cifar100'.")


# ---------- utilities ----------
class ShuffledDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


def split_data(dataset, worker_list=None, n_workers=2):
    if worker_list is None:
        worker_list = list(range(0, n_workers))
    idx = 0
    dic_single_datasets = {worker: [] for worker in worker_list}
    label_list, index_list, index_list_UUID = [], [], []
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


# ---------- VFL roles ----------
class Client:
    def __init__(self, client_id, dataset, model):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model.to(DEVICE)

    def forward(self, data):
        return self.model(data.to(DEVICE))


class Server:
    def __init__(self, model, criterion):
        self.model = model.to(DEVICE)
        self.criterion = criterion

    def update(self, n_workers, client_outputs, labels, train_mode, optimizer):
        gradients = None
        if train_mode:
            combined_output = torch.cat(client_outputs, 2)  # concat on feature dim
            server_output = self.model(combined_output)
            loss = self.criterion(server_output.to(DEVICE), labels.to(DEVICE)).to(DEVICE)
            loss.backward(retain_graph=True)
            gradients = [torch.autograd.grad(loss, output, retain_graph=True)[0] for output in client_outputs]
        else:
            self.model.eval()
            with torch.no_grad():
                t_combined_output = torch.cat(client_outputs, 2)
                server_output = self.model(t_combined_output)
                loss = self.criterion(server_output, labels)
        return loss, server_output, gradients


# ---------- train / test ----------
def train(epoch, client_instances, n_workers, client_optimizers, server, data_loaders, label_loader, serverOPT):
    start_time = time.time()
    client_times = []

    dataset_length = len(data_loaders[0].dataset)
    indices = torch.randperm(dataset_length).tolist()
    shuffled_data_datasets = [ShuffledDataset(dl.dataset, indices) for dl in data_loaders]
    shuffled_label_dataset = ShuffledDataset(label_loader.dataset, indices)
    shuffled_data_loaders = [DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
                             for dataset in shuffled_data_datasets]
    shuffled_label_loader = DataLoader(shuffled_label_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    for client in client_instances:
        client.model.train()

    losses, correct, total = [], 0, 0
    for data_tuple in zip(*shuffled_data_loaders, shuffled_label_loader):
        batch_data = data_tuple[:-1]
        labels = data_tuple[-1]

        output_List = []
        for i, client in enumerate(client_instances):
            t0 = time.time()
            output = client.forward(batch_data[i])
            client_times.append(time.time() - t0)
            output_List.append(output)

        client_outputs = [out.detach().requires_grad_(True) for out in output_List]
        loss, o_sum, client_gradients = server.update(n_workers, client_outputs, labels.to(DEVICE), True, None)

        for i in range(n_workers):
            client_optimizers[i].zero_grad()
            output_List[i].backward(client_gradients[i])
            client_optimizers[i].step()

        losses.append(loss.item())
        _, predicted = torch.max(o_sum.data, 1)
        total += labels.size(0)
        correct += (predicted.to(DEVICE) == labels.to(DEVICE)).sum().item()

    train_accuracy = 100.0 * correct / total if total else 0.0
    total_time = time.time() - start_time
    avg_client_time = sum(client_times) / len(client_times) if client_times else 0.0
    print(f"Epoch {epoch+1}, Train Acc: {train_accuracy:.2f}%")
    return losses, float(np.mean(losses)) if losses else 0.0, train_accuracy, avg_client_time, total_time


def test(n_workers, server, client_instances, val_loaders, t_local):
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, labels in zip(zip(*[dl for dl in val_loaders]), t_local):
            client_outputs = []
            batch_data = [d.to(DEVICE) for d in batch_data]
            labels = labels.to(DEVICE)
            for client, client_data in zip(client_instances, batch_data):
                client_outputs.append(client.forward(client_data))

            loss, output, _ = server.update(n_workers, client_outputs, labels, False, None)
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(t_local.dataset)
    accuracy = 100.0 * correct / total if total else 0.0
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return test_loss, accuracy, precision, recall, conf_matrix, f1_score


# ---------- orchestration ----------
def main(
    dataset: str,
    model: str,
    data_root: str,
    timestep: int,
    learning_rate: float,
    n_workers: int,
    leak_mem: float,
    img_size: int,
    momentum: float,
    weight_decay: float,
    num_classes_override: int = None,
):
    transform = build_transforms()
    trainset, valset, inferred_num_classes, csv_filename = load_dataset(dataset, data_root, transform)
    num_classes = num_classes_override if num_classes_override is not None else inferred_num_classes

    # int label lists for loaders
    labels = type(trainset)(data_root, download=True, train=True).targets[:len(trainset)]
    labels_val = type(valset)(data_root, download=True, train=False).targets[:len(valset)]

    img, _, _, _ = split_data(trainset, n_workers=n_workers)
    data_loaders = [DataLoader(img[w], batch_size=BATCH_SIZE, drop_last=True, shuffle=False) for w in range(n_workers)]
    img_test, _, _, _ = split_data(valset, n_workers=n_workers)
    val_loaders = [DataLoader(img_test[w], batch_size=BATCH_SIZE, drop_last=True, shuffle=False) for w in range(n_workers)]
    label_loader = DataLoader(labels, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    t_local = DataLoader(labels_val, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

    # compute vertical widths for each client
    base_size = img_size // n_workers
    remainder = img_size % n_workers
    widths = [base_size] * n_workers
    widths[-1] += remainder
    print("Vertical splits (width per client):", widths)

    # Build client models
    client_models, client_optimizers, client_instances = [], [], []
    if model == "sresnet":
        for i in range(n_workers):
            model = SResnet(n=2, nFilters=32, num_steps=timestep, leak_mem=leak_mem,
                            img_size=img_size, Y_img_size=widths[i], num_cls=num_classes,
                            boosting=False, poisson_gen=True)
            client_models.append(model)
    elif model == "vggsnn":
        for i in range(n_workers):
            model = VGGSNNClient(num_steps=timestep, leak_mem=leak_mem,
                                 X_img_size=img_size, Y_img_size=widths[i], num_cls=num_classes)
            client_models.append(model)
    else:
        raise ValueError("Unsupported model. Choose 'sresnet' or 'vggsnn'.")

    for i in range(n_workers):
        opt = torch.optim.SGD(client_models[i].parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        client_optimizers.append(opt)
        client_instances.append(Client(f'client{i}', data_loaders[i], client_models[i]))

    # Build server model
    if model == "sresnet":
        server_model = ServerM(nFilters=32, num_steps=timestep, leak_mem=leak_mem, img_size=img_size,
                               Y_img_size=widths[-1], num_cls=num_classes, boosting=False, poisson_gen=True,
                               n_clients=n_workers)
    else:  # vggsnn
        server_model = VGGSNNServer(num_steps=timestep, num_cls=num_classes, leak_mem=leak_mem)

    serverOPT = torch.optim.SGD(server_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    server = Server(server_model, torch.nn.CrossEntropyLoss())

    training_losses, mean_losses, test_losses = [], [], []
    accuracies, trainAccuracies = [], []
    best_accuracy, best_epoch_metrics = 0.0, {}
    avg_client_times, total_times = [], []

    for epoch in range(EPOCHS):
        tr_loss_list, mean_loss, tr_acc, avg_client_time, total_time = train(
            epoch, client_instances, n_workers, client_optimizers, server, data_loaders, label_loader, serverOPT
        )
        avg_client_times.append(avg_client_time)
        total_times.append(total_time)
        test_loss, accuracy, precision, recall, conf_matrix, f1_score = test(
            n_workers, server, client_instances, val_loaders, t_local
        )

        training_losses += tr_loss_list
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        trainAccuracies.append(tr_acc)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch_metrics = {
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'conf_matrix': conf_matrix
            }

    print(f"[{dataset} / {model}] Best accuracy at epoch {best_epoch_metrics['epoch']}: {best_accuracy:.2f}")
    print(f"Precision: {best_epoch_metrics['precision']:.4f}, Recall: {best_epoch_metrics['recall']:.4f}")
    print("Confusion Matrix:\n", best_epoch_metrics['conf_matrix'])

    with open(f"vfl_results_{dataset}_{model}.csv", mode='a', newline='') as file:
        w = csv.writer(file)
        if file.tell() == 0:
            w.writerow(['Metric', 'Value'])
        w.writerow(['Dataset', dataset])
        w.writerow(['model', model])
        w.writerow(['Number of Workers', n_workers])
        w.writerow(['Learning Rate', learning_rate])
        w.writerow(['T', timestep])
        w.writerow(['AVGTrainACC', float(np.mean(trainAccuracies)) if trainAccuracies else 0.0])
        w.writerow(['Best Accuracy', best_accuracy])
        w.writerow(['Final Accuracy', accuracies[-1] if accuracies else 0.0])
        w.writerow(['Average Accuracy', float(np.mean(accuracies)) if accuracies else 0.0])
        w.writerow(['Best Epoch', best_epoch_metrics['epoch']])
        w.writerow(['Precision', best_epoch_metrics['precision']])
        w.writerow(['Recall', best_epoch_metrics['recall']])
        w.writerow(['f1_score', best_epoch_metrics['f1_score']])
        conf_matrix_str = '\n'.join(['\t'.join([str(cell) for cell in row]) for row in best_epoch_metrics['conf_matrix']])
        w.writerow(['Confusion Matrix', conf_matrix_str])
        w.writerow(['Average Client Training Time (seconds)', float(np.mean(avg_client_times)) if avg_client_times else 0.0])
        w.writerow(['Total Training Time (seconds)', float(np.sum(total_times)) if total_times else 0.0])


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="VFL SNN on CIFAR-10/100 with vertical partitioning")
    p.add_argument("--dataset",  type=str, default="cifar100", choices=["cifar10", "cifar100"], help="Dataset")
    p.add_argument("--model", type=str, default="sresnet",  choices=["sresnet", "vggsnn"],  help="Model model")
    p.add_argument("--data-root", type=str, default=None, help="Datasets root (default from config)")
    # If you pass None for these, defaults are pulled from CFG[dataset]['models'][model]
    p.add_argument("--timestep", type=int,   default=None, help="Number of SNN time steps")
    p.add_argument("--lr",       type=float, default=None, help="Learning rate")
    p.add_argument("--momentum", type=float, default=None, help="SGD momentum")
    p.add_argument("--weight-decay", type=float, default=None, help="SGD weight decay")
    p.add_argument("--clients",  type=int,   default=2,    help="Number of vertical clients")
    p.add_argument("--leak-mem", type=float, default=DEFAULT_LEAK_MEM, help="Leak membrane (SNN)")
    p.add_argument("--img-size", type=int,   default=IMG_SIZE, help="Input image size (square)")
    p.add_argument("--num-classes", type=int, default=None, help="Override number of classes")
    return p.parse_args()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(DEVICE)

    args = parse_args()
    cfg = CFG[args.dataset]["models"][args.model]

    data_root = args.data_root or DATA_ROOT
    timestep = args.timestep if args.timestep is not None else cfg["timestep"]
    lr       = args.lr       if args.lr       is not None else cfg["lr"]
    momentum = args.momentum if args.momentum is not None else cfg["momentum"]
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg["weight_decay"]

    torch.cuda.empty_cache()

    main(
        dataset=args.dataset,
        model=args.model,
        data_root=data_root,
        timestep=timestep,
        learning_rate=lr,
        n_workers=args.clients,
        leak_mem=args.leak_mem,
        img_size=args.img_size,
        momentum=momentum,
        weight_decay=weight_decay,
        num_classes_override=args.num_classes,
    )

