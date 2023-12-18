import os
import glob
import sys
import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from PIL import Image

from basicsr.archs.raft_arch import RAFT_SR
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.maskflownet_arch import MaskFlownet_S

from basicsr.utils import flow_to_image


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img,
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (
            y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature,
              flow,
              mask=False,
              mode='bilinear',
              padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature,
                           grid,
                           mode=mode,
                           padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow,
                                       bwd_flow,
                                       alpha=0.01,
                                       beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow
    # (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


@torch.no_grad()
def get_warped_and_mask(flow_model,
                        image1,
                        image2,
                        image3=None,
                        pixel_consistency=False):
    if image3 is None:
        image3 = image1
    padder = InputPadder(image1.shape, padding_factor=8)
    image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
    results_dict = flow_model(image1,
                              image2,
                              attn_splits_list=[2],
                              corr_radius_list=[-1],
                              prop_radius_list=[-1],
                              pred_bidir_flow=True)
    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
    fwd_occ, bwd_occ = forward_backward_consistency_check(
        fwd_flow, bwd_flow)  # [1, H, W] float
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(
            bwd_occ + (abs(image2 - warped_image1).mean(dim=1) > 255 * 0.25).float(), 0, 1
        ).unsqueeze(0)
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ, bwd_flow


def compute_flow(imgs, flownet):
    """Compute optical flow using SPyNet for feature warping.

    Args:
        lrs (tensor): Input LR images with shape (n, t, c, h, w)

    Return:
        tuple(Tensor): Optical flow. 
            'flows_forward' corresponds to the flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for backward-time propagation (current to next).
    """
    n, t, c, h, w = imgs.size()
    imgs_1 = imgs[:, :-1, :, :, :].reshape(-1, c, h, w)  # former
    imgs_2 = imgs[:, 1:, :, :, :].reshape(-1, c, h, w)   # latter
    
    # forward flow, for backward propagation
    flows_backward = flownet(imgs_1, imgs_2).view(n, t - 1, 2, h, w)
    # backward flow, for forward propagation
    flows_forward = flownet(imgs_2, imgs_1).view(n, t - 1, 2, h, w)

    return flows_forward, flows_backward


def compute_flow_1b1(imgs, flownet):
    """Compute optical flow using SPyNet for feature warping.

    Args:
        lrs (tensor): Input LR images with shape (n, t, c, h, w)

    Return:
        tuple(Tensor): Optical flow. 
            'flows_forward' corresponds to the flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for backward-time propagation (current to next).
    """
    n, t, c, h, w = imgs.size()
    imgs_1 = imgs[:, :-1, :, :, :]#.reshape(-1, c, h, w)  # former
    imgs_2 = imgs[:, 1:, :, :, :]#.reshape(-1, c, h, w)   # latter
    
    flows_backward = list()
    flows_forward = list()
    num_flows = imgs.size(1) - 1
    for i in range(num_flows):
        flo_for_bwd_prop = flownet(imgs_1[:, i, :, :, :], imgs_2[:, i, :, :, :])
        flows_backward.append(flo_for_bwd_prop)
        flo_for_fwd_prop = flownet(imgs_2[:, i, :, :, :], imgs_1[:, i, :, :, :])
        flows_forward.append(flo_for_fwd_prop)
    # # forward flow, for backward propagation
    # flows_backward = flownet(imgs_1, imgs_2).view(n, t - 1, 2, h, w)
    # # backward flow, for forward propagation
    # flows_forward = flownet(imgs_2, imgs_1).view(n, t - 1, 2, h, w)
    # flows_backward = torch.stack(flows_backward, dim=1)
    # flows_forward = torch.stack(flows_forward, dim=1)

    return flows_forward, flows_backward


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def get_flow_imgs(src_root, dst_root, model, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        videos = sorted(glob.glob(os.path.join(src_root, '*')))
        for video in videos:
            print(f'Processing: {os.path.basename(video)}')
            images = sorted(glob.glob(os.path.join(video, '*.png')))
            for imfile1, imfile2 in zip(images[:-1], images[1:]):
                image1 = load_image(imfile1, device) / 255.  # [1, C, H, W]
                image2 = load_image(imfile2, device) / 255.  # [1, C, H, W]

                flow = model(image1, image2)
                flow = flow[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                
                mkdir(os.path.join(dst_root, os.path.basename(video)))
                flow_img = flow_to_image(flow)
                filename = os.path.join(
                    dst_root, 
                    os.path.basename(video),
                    '{}.png'.format(os.path.basename(imfile1.split('.')[0]))
                )
                # writeFlow(filename=filename, uv=flow)
                cv2.imwrite(filename, flow_img[:, :, [2, 1, 0]])
