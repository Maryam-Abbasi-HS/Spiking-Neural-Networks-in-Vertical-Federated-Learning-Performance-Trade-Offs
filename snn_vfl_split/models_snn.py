

# models_snn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

# ------------- Shared SNN utilities -------------
class SurrogateBPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).to(input.device)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(inp), 0, 0)
        return grad


def poisson_gen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).to(inp.device)
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


# ------------- S-ResNet client (from your code) -------------
class SResnet(nn.Module):
    def __init__(
        self,
        n: int,
        nFilters: int,
        num_steps: int,
        leak_mem: float = 0.95,
        img_size: int = 32,
        Y_img_size: int = 32,
        num_cls: int = 10,     # default; overridden by caller based on dataset
        boosting: bool = False,
        poisson_gen: bool = True,
    ):
        super(SResnet, self).__init__()

        self.n = n
        self.Y_img_size = Y_img_size
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.poisson_gen = poisson_gen
        self.boost = nn.AvgPool1d(10, 10) if boosting else False

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        # Stem
        self.conv1 = nn.Conv2d(3, self.nFilters, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        # 3 blocks, 2*n layers total per block; first layer of blocks 2/3 downsamples
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
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=bias_flag,
                    )
                )
                self.bntt_list.append(
                    nn.ModuleList(
                        [
                            nn.BatchNorm2d(
                                self.nFilters * (2 ** block), eps=1e-4, momentum=0.1, affine=affine_flag
                            )
                            for _ in range(self.batch_num)
                        ]
                    )
                )

        # 1x1 residual downsamplers for the starts of blocks 2 and 3
        self.conv_resize_1 = nn.Conv2d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0, bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters * 2, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0, bias=bias_flag)
        self.resize_bn_2 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters * 4, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)]
        )

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        # Kept to match your original structure (unused in client head, server does final FC)
        self.fc = nn.Linear(self.nFilters * 4, self.num_cls * 10 if self.boost else self.num_cls, bias=bias_flag)

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])
        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1, self.resize_bn_2])

        # Turn off bias of BNTT (as in your code)
        for bn_temp in self.bntt_list:
            bn_temp.bias = None

        # Initialize thresholds + weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Returns: [num_steps, batch, features] where features = nFilters*4 after GAP.
        Uses lazy membrane buffers that adopt the exact spatial sizes produced by convs,
        so odd widths from vertical splits are handled safely.
        """
        device = inp.device
        batch_size = inp.size(0)

        # Lazy per-layer membranes — exact shapes are known after conv+BN
        mem_conv_list = [None for _ in range(len(self.conv_list))]

        outlist = []
        for t in range(self.num_steps):
            out_prev = poisson_gen(inp) if self.poisson_gen else inp

            index_1x1 = 0
            skip = None  # residual state

            for i in range(len(self.conv_list)):
                # conv + BNTT for this timestep
                conv_out = self.conv_list[i](out_prev)
                bn_out = self.bntt_list[i][t](conv_out)

                # allocate membrane lazily to the real shape produced by conv/bn
                if mem_conv_list[i] is None:
                    mem_conv_list[i] = torch.zeros_like(bn_out, device=device)

                # integrate & fire
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + bn_out
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)

                # residual connections
                if i > 0 and i % 2 == 0:
                    # at starts of blocks 2 and 3, downsample the skip path via 1x1 conv
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                # reset spiking neurons
                rst = torch.zeros_like(mem_conv_list[i], device=device)
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst

                out_prev = out.clone()

                # after last conv layer, global average pool
                if i == len(self.conv_list) - 1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            # flatten features of this timestep
            out_prev = out_prev.reshape(batch_size, -1)
            outlist.append(out_prev)

        # [num_steps, batch, nFilters*4]
        return torch.stack(outlist)


class ServerM(nn.Module):
    """Server head for S-ResNet clients; feature per client is fixed (AdaptiveAvgPool -> 1x1)."""
    def __init__(
        self,
        nFilters: int,
        num_steps: int,
        leak_mem: float = 0.95,
        img_size: int = 32,
        Y_img_size: int = 32,
        num_cls: int = 100,
        boosting: bool = False,
        poisson_gen: bool = True,
        n_clients: int = 2,
    ):
        super(ServerM, self).__init__()
        self.nFilters = nFilters
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.poisson_gen = poisson_gen
        self.boost = nn.AvgPool1d(10, 10) if boosting else False
        self.n_clients = n_clients

        in_features = self.nFilters * 4 * self.n_clients
        self.fc = nn.Linear(in_features, self.num_cls * 10 if self.boost else self.num_cls, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [num_steps, batch, in_features]
        batch_size = z.shape[1]
        device = z.device
        mem_fc = torch.zeros(batch_size, self.num_cls, device=device)
        for t in range(self.num_steps):
            inp_t = z[t]
            if self.boost:
                mem_fc = mem_fc + self.boost(self.fc(inp_t).unsqueeze(1)).squeeze(1)
            else:
                mem_fc = mem_fc + self.fc(inp_t)
        return mem_fc / self.num_steps


# ------------- VGG-style SNN client/server (from your ClientM3/ServerM) -------------
class VGGSNNClient(nn.Module):
    """Your ClientM3, device-agnostic, returns [num_steps, batch, features]."""
    def __init__(self, num_steps=20, leak_mem=0.95, X_img_size=32, Y_img_size=32, num_cls=100):
        super().__init__()
        self.batch_num = num_steps
        self.X_img_size = X_img_size
        self.Y_img_size = Y_img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for _ in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        device = inp.device
        B = inp.size(0)
        mem = [
            torch.zeros(B, 64,  self.X_img_size,          self.Y_img_size,          device=device),
            torch.zeros(B, 64,  self.X_img_size,          self.Y_img_size,          device=device),
            torch.zeros(B, 128, self.X_img_size // 2,     self.Y_img_size // 2,     device=device),
            torch.zeros(B, 128, self.X_img_size // 2,     self.Y_img_size // 2,     device=device),
            torch.zeros(B, 256, self.X_img_size // 4,     self.Y_img_size // 4,     device=device),
            torch.zeros(B, 256, self.X_img_size // 4,     self.Y_img_size // 4,     device=device),
            torch.zeros(B, 256, self.X_img_size // 4,     self.Y_img_size // 4,     device=device),
        ]

        outputs = []
        for t in range(self.num_steps):
            out_prev = poisson_gen(inp)
            for i, (conv, bns) in enumerate(zip(self.conv_list, self.bntt_list)):
                mem[i] = self.leak_mem * mem[i] + bns[t](conv(out_prev))
                mem_thr = (mem[i] / conv.threshold) - 1.0
                out = self.spike_fn(mem_thr)

                rst = torch.zeros_like(mem[i], device=device)
                rst[mem_thr > 0] = conv.threshold
                mem[i] = mem[i] - rst
                out_prev = out.clone()

                pool = self.pool_list[i]
                if pool is not False:
                    out = pool(out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(B, -1)
            outputs.append(out_prev)

        return torch.stack(outputs)  # [num_steps, batch, features]

class VGGSNNServer(nn.Module):
    """
    Server head for VGG-style clients.
    Uses LazyLinear so it can infer the concatenated input width (depends on client splits).
    """
    def __init__(self, num_steps=20, num_cls=100, leak_mem=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.num_cls = num_cls
        self.leak_mem = leak_mem
        self.spike_fn = SurrogateBPFunction.apply

        # fc1 infers in_features at first forward; outputs 1024 features
        self.fc1 = nn.LazyLinear(1024, bias=False)
        self.bntt_fc = nn.ModuleList(
            [nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=True) for _ in range(self.num_steps)]
        )
        self.fc2 = nn.Linear(1024, self.num_cls, bias=False)

     
        for bn in self.bntt_fc:
            bn.bias = None

        # Initialize thresholds on modules; ONLY init weights when materialized
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
              
                if not isinstance(m.weight, UninitializedParameter):
                    nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [num_steps, batch, concat_features]
        B = z.shape[1]
        device = z.device
        mem_fc1 = torch.zeros(B, 1024, device=device)
        mem_fc2 = torch.zeros(B, self.num_cls, device=device)

        for t in range(self.num_steps):
            inp_t = z[t]
            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(inp_t))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)

            rst = torch.zeros_like(mem_fc1, device=device)
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        return mem_fc2 / self.num_steps



