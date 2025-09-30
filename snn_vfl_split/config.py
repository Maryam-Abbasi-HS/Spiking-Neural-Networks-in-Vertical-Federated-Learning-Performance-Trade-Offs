
# config.py
import torch

# Core runtime
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 40
IMG_SIZE = 32
DATA_ROOT = "CIFAR_DATA"

# Per-dataset + per-backbone defaults
# You can override any of these via CLI flags in vfl_train_test.py
CFG = {
    "cifar10": {
        "num_classes": 10,
        "backbones": {
            "sresnet":  {"timestep": 32, "lr": 0.05,   "momentum": 0.9,  "weight_decay": 5e-4},
            "vggsnn":   {"timestep": 32, "lr": 0.05,   "momentum": 0.9,  "weight_decay": 5e-4},
        },
    },
    "cifar100": {
        "num_classes": 100,
        "backbones": {
            "sresnet":  {"timestep": 32, "lr": 0.02684, "momentum": 0.95, "weight_decay": 1e-4},
            "vggsnn":   {"timestep": 32, "lr": 0.02684, "momentum": 0.95, "weight_decay": 1e-4},
        },
    },
}

# SNN membrane leak default (can be overridden)
DEFAULT_LEAK_MEM = 0.99

