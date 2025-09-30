# config.py
import torchvision.transforms as transforms

DEFAULTS = {
    "batch_size": 128,
    "epochs": 40,
    "img_size": 32,
    "leak_mem": 0.99,
    "momentum": 0.95,
    "weight_decay": 1e-4,
    "timesteps": 32,
    "learning_rate": 0.1,
    "dropout": 0.5,  # only used by VGG-SNN
}

# Normalization used for both CIFAR-10 and CIFAR-100
_NORM = transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))

DATASETS = {
    "cifar10": {
        "num_classes": 10,
        "transform": transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((32, 32)),
                                         _NORM]),
     
        "learning_rate": 0.1,
        "timesteps": 32,
        "leak_mem": 0.99,
    },
    "cifar100": {
        "num_classes": 100,
        "transform": transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((32, 32)),
                                         _NORM]),
       
        "learning_rate": 0.03,
        "timesteps": 32,
        "leak_mem": 0.99,
    },
}


MODELS = {
    "vggsnn": {
        # head width for the first FC in VGG-style SNN client
        "fc_dim": 1024,
    },
    "sresnet": {
        "n": 2,
        "nFilters": 32,
    },
}

