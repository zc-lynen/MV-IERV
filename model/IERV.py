import torch
import torch.nn as nn
from .model_utils import ActivationLayer, NormLayer
from .model_utils import weight_quantize_fn, conv2d_quantize_fn


class ConvBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        Conv2d = conv2d_quantize_fn(kargs['qbit'])
        ngf, new_ngf = kargs['ngf'], kargs['new_ngf']

        self.conv = Conv2d(ngf, new_ngf * kargs['stride'] * kargs['stride'], 3, 1, 1, bias=kargs['bias'])
        self.up_scale = nn.PixelShuffle(kargs['stride'])
        self.norm = NormLayer(kargs['norm'], new_ngf)
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x, emb):
        return self.act(self.norm(self.up_scale(self.conv(x))+emb))


class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        # quantization
        Conv2d = conv2d_quantize_fn(kargs['qbit'])
        self.quantize_fn = weight_quantize_fn(kargs['qbit'])

        # hyperparameters
        self.t_dim = kargs['t_dim']
        self.v_dim = kargs['v_dim']
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fea_hw_dim'].split('_')]
        self.strides = kargs['stride_list']

        # temporal feature grid
        self.input_t_grids = nn.ParameterList()
        grid_reso = [self.t_dim, self.t_dim // 2]
        for g_r in grid_reso:
            self.input_t_grids.append(nn.Parameter(nn.init.xavier_uniform_(
                torch.empty(g_r, self.fc_dim // len(grid_reso), self.fc_h, self.fc_w))))

        # view feature grid
        self.input_v_grids = nn.ParameterList()
        self.input_v_grids.append(nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.v_dim, self.fc_dim, self.fc_h, self.fc_w)))) # 2*self.fc_dim

        # conv layers
        self.global_grids = nn.ParameterList()
        self.norm_layers, self.conv_layers, self.head_layers, self.addi_layers = [nn.ModuleList() for _ in range(4)]
        ngf = self.fc_dim

        for i, stride in enumerate(self.strides):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            # global grid of time in one view
            self.global_grids.append(nn.Parameter(nn.init.xavier_uniform_(
                torch.empty(self.v_dim, new_ngf, stride, stride))))

            # norm
            self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))

            # conv
            self.conv_layers.append(ConvBlock(
                ngf=ngf, new_ngf=new_ngf, bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], qbit=kargs['qbit'], stride=stride))

            ngf = new_ngf

            # head layer
            head_layer = None
            addi_layer = None

            if i == len(self.strides) - 1:
                head_layer = Conv2d(ngf, 3+2, 1, 1, 0, bias=kargs['bias'])
                addi_layer = Conv2d(ngf, 3, 1, 1, 0, bias=kargs['bias'])

            self.head_layers.append(head_layer)
            self.addi_layers.append(addi_layer)

        self.sigmoid = kargs['sigmoid']

    def fuse_tv(self, t, v):
        f_dim = v.shape[1] // 2
        gamma = v[:, :f_dim]
        beta = v[:, f_dim:]
        out = t * gamma + beta

        return out

    def get_grid(self, input_t, grids):
        cur_grid = None
        scale = 1
        for g_idx, grid in enumerate(grids):
            grid = self.quantize_fn(grid)
            inp = torch.floor(input_t / scale).long()
            g = grid[inp]

            if g_idx == 0:
                cur_grid = g
            else:
                cur_grid = torch.cat([cur_grid, g], 1)
            scale *= 2

        return cur_grid

    def forward(self, time, view):
        emb_t = self.get_grid(time, self.input_t_grids)
        emb_v = self.get_grid(view, self.input_v_grids)
        # output = self.fuse_tv(emb_t, emb_v)
        output = emb_t + emb_v

        out_list = []
        for conv_layer, head_layer, addi_layer, global_grid, norm_layer \
                in zip(self.conv_layers, self.head_layers, self.addi_layers, self.global_grids, self.norm_layers):
            global_grid = self.quantize_fn(global_grid)
            _, _, rows, cols = output.size()
            emb_global = global_grid[view.long()].repeat(1, 1, rows, cols)
            
            output = norm_layer(output)
            output = conv_layer(output, emb_global)

            if addi_layer is not None:
                addi_out = addi_layer(output)
                out_list.append(addi_out)

            if head_layer is not None:
                img_out = head_layer(output)
                out_list.append(img_out)

        # normalize the independent frame to [0,1]
        out_list[1][:,0:3,:,:] = torch.sigmoid(out_list[1][:,0:3,:,:]) if self.sigmoid else (torch.tanh(out_list[1][:,0:3,:,:]) + 1) * 0.5

        return out_list