import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed_b, pe_embed_l):
        super(PositionalEncoding, self).__init__()
        if pe_embed_b == 0:
            self.embed_length = 1
            self.pe_embed = False
        else:
            self.lbase = float(pe_embed_b)
            self.levels = float(pe_embed_l)
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels
            self.pe_embed = True
    
    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        if self.pe_embed is False:
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(30 * input)  # see SIREN paper for the factor 30


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    elif act_type == 'non':
        act_layer = nn.Identity()
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


### frame warping function:
def get_grid(flow):
    m, n = flow.shape[-2:]
    shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
    shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

    grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
    workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

    flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

    return flow_grid


# warping function
def resample(feats, flow):
    scale_factor = float(feats.shape[-1]) / flow.shape[-1]
    flow = torch.nn.functional.interpolate(
        flow, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    flow = flow * scale_factor
    flow_grid = get_grid(flow)
    warped_feats = F.grid_sample(feats, flow_grid, mode="bilinear", padding_mode="border")
    return warped_feats


### quantization function: Any-Precision Deep Neural Networks (AAAI 2021)
class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**(k-1) - 1)
        out = torch.floor(torch.abs(input) * n) / n
        out = out*torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            weight_q = x
        else:
            weight = torch.tanh(x)
            weight_q = qfn.apply(weight, self.wbit)
        return weight_q


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)


def conv2d_quantize_fn(bit):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.w_bit = bit
            self.quantize_fn = weight_quantize_fn(self.w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            bias_q = self.quantize_fn(self.bias)
            return F.conv2d(input, weight_q, bias_q, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_