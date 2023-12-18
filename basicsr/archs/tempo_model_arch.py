import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicsr.archs.spynet_arch import SpyNet


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


class CouplePropModule(nn.Module):
    """Couple Propagation Module.

    Args:
        num_ch (int): Number of input channels. Default: 4.
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15.
    """

    def __init__(self,
                 num_ch=4,
                 num_feat=64,
                 num_block=5):
        super().__init__()
        
        self.num_ch = num_ch
        self.num_feat = num_feat

        # propagation
        self.backward_trunk = ConvResidualBlocks(1 * num_feat + num_ch, num_feat, num_block)
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)

        self.forward_trunk = ConvResidualBlocks(2 * num_feat + num_ch, num_feat, num_block)
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)

        # reconstruction
        self.conv_last = nn.Conv2d(num_feat, num_ch, 3, 1, 1)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, flows):
        b, n, _, h_input, w_input = x.size()

        h, w = x.shape[3:]

        # compute flow and keyframe features
        flows_forward, flows_backward = flows

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            out = self.conv_last(feat_prop)
            out += x_i
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class CouplePropModuleWithFlowNet(nn.Module):
    """Couple Propagation Module.

    Args:
        num_ch (int): Number of input channels. Default: 4.
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 5.
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self,
                 num_ch=4,
                 num_feat=64,
                 num_block=5,
                 spynet_path=None):
        super().__init__()
        
        self.num_ch = num_ch
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(1 * num_feat + num_ch, num_feat, num_block)
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)

        self.forward_trunk = ConvResidualBlocks(2 * num_feat + num_ch, num_feat, num_block)
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)

        # reconstruction
        self.conv_last = nn.Conv2d(num_feat, num_ch, 3, 1, 1)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x, lrs):
        b, n, _, h_input, w_input = x.size()

        h, w = x.shape[3:]

        # compute flow
        flows_forward, flows_backward = self.get_flow(lrs)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            out = self.conv_last(feat_prop)
            out += x_i
            out_l[i] = out

        return torch.stack(out_l, dim=1)


if __name__ == '__main__':

    device = torch.device('cuda')
    
    # inp = torch.rand(1, 10, 4, 64, 64).to(device)
    # net = CouplePropModule(num_ch=4, num_feat=64, num_block=5).to(device)
    # flows = [
    #     torch.rand(1, 9, 2, 64, 64).to(device),
    #     torch.rand(1, 9, 2, 64, 64).to(device)
    # ]
    # out = net(inp, flows)
    # print(out.shape)

    inp = torch.rand(1, 10, 4, 64, 64).to(device)
    lrs = torch.rand(1, 10, 3, 64, 64).to(device)
    net = CouplePropModuleWithFlowNet(num_ch=4, num_feat=64, num_block=5).to(device)
    out = net(inp, lrs)
    print(out.shape)