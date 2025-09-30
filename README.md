# Spiking-Neural-Networks-in-Vertical-Federated-Learning-Performance-Trade-Offs


This repository contains the official code implementation of the paper **"Spiking Neural Networks in Vertical Federated Learning: Performance Trade-Offs"**.  

Our goal is to investigate the trade-offs in accuracy, efficiency, and communication cost when applying **Spiking Neural Networks (SNNs)** in VFL setups.

# Features
- Vertical data partitioning of CIFAR-10 and CIFAR-100 datasets.  
- Support for **VGG-SNN** and **Spiking ResNet** with model spiliting and without model spiliting.
# RUN
   - # dependencies 
 torch torchvision numpy scikit-learn
 
  - # Arguments:

--dataset : cifar10 | cifar100 (default: cifar100)

--model : vggsnn | sresnet (default: vggsnn)

--clients : number of vertical clients (default: 2)

--epochs : override number of epochs

--batch_size : override batch size

--timesteps : override number of SNN timesteps

--lr : override learning rate

--momentum : override SGD momentum

--weight_decay : override weight decay

--leak_mem : override SNN membrane leak parameter

- # Default (CIFAR-100, VGG-SNN, 2 clients):
python vfl_train_test.py
- # Common options:
python vfl_train_test.py \
  --dataset cifar10            # or cifar100 (default)
  --model sresnet              # or vggsnn (default)
  --clients 4                  # number of vertical clients (default 2)
  --epochs 50                  # override epochs
  --batch_size 128             # override batch size
  --timesteps 8                # SNN timesteps
  --lr 0.01 --momentum 0.9 --weight_decay 1e-4 --leak_mem 0.95
  
# Ackonwledgements: 

The code is **adapted from [HFL-SNN (Horizontal Federated Learning with Spiking Neural Networks)](https://github.com/Intelligent-Computing-Lab-Panda/FedSNN)**
