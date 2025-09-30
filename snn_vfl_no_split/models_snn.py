# models_snn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SurrogateBPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input, device=input.device)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(inp), 0, 0)
        return grad


def poisson_gen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp, device=inp.device)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


# -------------------------
# VGG-style SNN (client)
# -------------------------
class VGGClientSNN(nn.Module):
    def __init__(self, timesteps=20, leak_mem=0.95, X_img_size=32, Y_img_size=32,
                 num_cls=100, fc_dim=1024, dropout=0.5):
        super().__init__()
        self.X_img_size = X_img_size
        self.Y_img_size = Y_img_size
        self.num_cls = num_cls
        self.timesteps = timesteps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps
        self.dropout = nn.Dropout(dropout)

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag)
                                    for _ in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        # flatten -> fc -> logits for this client
        flat_dim = (X_img_size // 8) * (Y_img_size // 8) * 256
        self.fc1 = nn.Linear(flat_dim, fc_dim, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(fc_dim, eps=1e-4, momentum=0.1, affine=affine_flag)
                                      for _ in range(self.batch_num)])
        self.fc2 = nn.Linear(fc_dim, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        # Initialize thresholds/weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):
        batch_size = inp.size(0)
        device = inp.device

        # conv mem
        mem_conv1 = torch.zeros(batch_size, 64, self.X_img_size, self.Y_img_size, device=device)
        mem_conv2 = torch.zeros_like(mem_conv1, device=device)
        mem_conv3 = torch.zeros(batch_size, 128, self.X_img_size // 2, self.Y_img_size // 2, device=device)
        mem_conv4 = torch.zeros_like(mem_conv3, device=device)
        mem_conv5 = torch.zeros(batch_size, 256, self.X_img_size // 4, self.Y_img_size // 4, device=device)
        mem_conv6 = torch.zeros_like(mem_conv5, device=device)
        mem_conv7 = torch.zeros_like(mem_conv5, device=device)
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        # fc mem
        mem_fc1 = torch.zeros(batch_size, self.bntt_fc[0].num_features, device=device)
        mem_fc2 = torch.zeros(batch_size, self.num_cls, device=device)

        for t in range(self.timesteps):
            spike_inp = poisson_gen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)

                rst = torch.zeros_like(mem_conv_list[i], device=device)
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()

                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out = self.dropout(out)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)
            out_prev = self.dropout(out_prev)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1, device=device)
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps
        return out_voltage


# -------------------------
# S-ResNet (client)
# -------------------------
class SResnetClient(nn.Module):
    def __init__(self, n=2, nFilters=32, num_steps=20, leak_mem=0.95,
                 img_size=32, Y_img_size=32, num_cls=10, boosting=False, use_poisson=True):
        super().__init__()
        self.n = n
        self.Y_img_size = Y_img_size
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.use_poisson = use_poisson
        self.boost = nn.AvgPool1d(10, 10) if boosting else False

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        self.conv1 = nn.Conv2d(3, self.nFilters, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag)
             for _ in range(self.batch_num)]
        )

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        # 3 blocks; first layer of blocks 2 & 3 downsamples (stride=2)
        for block in range(3):
            for layer in range(2 * n):
                if block != 0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0
                self.conv_list.append(
                    nn.Conv2d(
                        self.nFilters * (2 ** (block + prev_nFilters)),
                        self.nFilters * (2 ** block),
                        kernel_size=3, stride=stride, padding=1, bias=bias_flag
                    )
                )
                self.bntt_list.append(
                    nn.ModuleList(
                        [nn.BatchNorm2d(self.nFilters * (2 ** block), eps=1e-4, momentum=0.1, affine=affine_flag)
                         for _ in range(self.batch_num)]
                    )
                )

        # 1x1 residual downsamplers for block starts
        self.conv_resize_1 = nn.Conv2d(self.nFilters,     self.nFilters * 2, kernel_size=1, stride=2, padding=0, bias=bias_flag)
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0, bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters * 2, eps=1e-4, momentum=0.1, affine=affine_flag)
             for _ in range(self.batch_num)]
        )
        self.resize_bn_2 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters * 4, eps=1e-4, momentum=0.1, affine=affine_flag)
             for _ in range(self.batch_num)]
        )
        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])
        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1, self.resize_bn_2])

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.nFilters * 4, self.num_cls * 10 if self.boost else self.num_cls, bias=bias_flag)

        # Disable BN bias (matching your original)
        for bn_list in self.bntt_list:
            bn_list.bias = None

        # Init thresholds + weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):
        device = inp.device
        B = inp.size(0)

        # lazy membranes: real shapes known only after conv -> BN (handles odd widths)
        mem_conv_list = [None for _ in range(len(self.conv_list))]
        mem_fc = torch.zeros(B, self.num_cls, device=device)

        for t in range(self.num_steps):
            out_prev = poisson_gen(inp) if self.use_poisson else inp

            index_1x1 = 0
            skip = None
            for i in range(len(self.conv_list)):
                conv_out = self.conv_list[i](out_prev)
                bn_out = self.bntt_list[i][t](conv_out)

                if mem_conv_list[i] is None:
                    mem_conv_list[i] = torch.zeros_like(bn_out, device=device)

                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + bn_out
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)

                if i > 0 and i % 2 == 0:
                    # start of blocks 2 and 3 → downsample skip
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                rst = torch.zeros_like(mem_conv_list[i], device=device)
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst

                out_prev = out.clone()

                if i == len(self.conv_list) - 1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            # flatten and accumulate logits
            out_prev = out_prev.reshape(B, -1)
            if self.boost:
                mem_fc = mem_fc + self.boost(self.fc(out_prev).unsqueeze(1)).squeeze(1)
            else:
                mem_fc = mem_fc + self.fc(out_prev)

        return mem_fc / self.num_steps


# -------------------------
# Simple VFL wrappers
# -------------------------
class Client:
    def __init__(self, client_id, dataset, model: nn.Module):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model.to(DEVICE)

    def forward(self, data):
        return self.model(data.to(DEVICE))


class ServerNoSplit:
    """
    No server model. Aggregates client logits by summation
    and computes the loss for backprop to clients.
    """
    def __init__(self, criterion):
        self.criterion = criterion

    def update(self, n_workers, client_outputs, labels, train_mode: bool):
        # Sum logits from clients
        sum_out = torch.stack(client_outputs).sum(dim=0)
        if train_mode:
            loss = self.criterion(sum_out, labels)
            gradients = [torch.autograd.grad(loss, out, retain_graph=True)[0] for out in client_outputs]
            return loss, sum_out, gradients
        else:
            with torch.no_grad():
                loss = self.criterion(sum_out, labels)
            return loss, sum_out, [None] * n_workers

