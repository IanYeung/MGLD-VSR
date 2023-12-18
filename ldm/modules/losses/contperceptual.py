import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

from basicsr.archs.arch_util import flow_warp, resize_flow
from basicsr.archs.spynet_arch import SpyNet
# from .focal_frequency_loss import FocalFrequencyLoss
from einops import repeat, rearrange
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from scripts.util_flow import forward_backward_consistency_check, get_warped_and_mask
from ldm.util import instantiate_from_config


def l1_diff(x, y, t):
    bt, _, _, _ = x.shape
    x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
    y = rearrange(y, '(b t) c h w -> b t c h w', t=t)
    diff_x = x[:, :-1, :, :, :] - x[:, 1:, :, :, :]
    diff_y = y[:, :-1, :, :, :] - y[:, 1:, :, :, :]
    out = rearrange(diff_x - diff_y, 'b t c h w -> (b t) c h w')
    return torch.abs(out)


def compute_flow(imgs, flownet):
    """Compute optical flow using flow estimation network for warping.

    Args:
        imgs (tensor): Input images with shape (n, t, c, h, w)

    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = imgs.size()
    imgs_1 = imgs[:, :-1, :, :, :].reshape(-1, c, h, w)  # former
    imgs_2 = imgs[:, 1:, :, :, :].reshape(-1, c, h, w)   # latter
    
    # forward flow, for backward propagation
    fwd_flows = flownet(imgs_1, imgs_2).view(n, t - 1, 2, h, w)
    # backward flow, for forward propagation
    bwd_flows = flownet(imgs_2, imgs_1).view(n, t - 1, 2, h, w)

    return fwd_flows, bwd_flows


def swc_loss(hr_frms, gt_frms, t, flownet, w=3, mask=True):
    bt, _, _, _ = hr_frms.shape
    # compute weight
    edge_maps = K.filters.sobel(gt_frms).detach()
    weight_maps = 1 + w * edge_maps
    # (b t) c h w -> b t c h w
    hr_frms = rearrange(hr_frms, '(b t) c h w -> b t c h w', t=t)
    gt_frms = rearrange(gt_frms, '(b t) c h w -> b t c h w', t=t)
    weight_maps = rearrange(weight_maps, '(b t) c h w -> b t c h w', t=t)
    # compute flows
    # fwd_flows: forward flow, for backward propagation
    # bwd_flows: backward flow, for forward propagation
    fwd_flows, bwd_flows = compute_flow(gt_frms, flownet)
    
    fwd_occ_list, bwd_occ_list = list(), list()
    for i in range(t-1):
        fwd_flow, bwd_flow = fwd_flows[:, i, :, :, :], bwd_flows[:, i, :, :, :]
        fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)
        fwd_occ_list.append(fwd_occ.unsqueeze_(1))
        bwd_occ_list.append(bwd_occ.unsqueeze_(1))
        fwd_occs = torch.stack(fwd_occ_list, dim=1)
        bwd_occs = torch.stack(bwd_occ_list, dim=1)
    fwd_occs = torch.stack(fwd_occ_list, dim=1)
    bwd_occs = torch.stack(bwd_occ_list, dim=1)

    loss_b = 0
    curr_warp = torch.zeros_like(hr_frms[:, -1, :, :, :])
    # backward propagation
    for i in range(t - 1, -1, -1):
        curr = hr_frms[:, i, :, :, :]
        if i < t - 1:  # no warping required for the last timestep
            flow = fwd_flows[:, i, :, :, :]
            curr_warp = flow_warp(curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_b += F.l1_loss(
                weight_maps[:, i, :, :, :] * (1 - fwd_occs[:, i, :, :, :]) * prev, 
                weight_maps[:, i, :, :, :] * (1 - fwd_occs[:, i, :, :, :]) * curr
            )
        prev = curr_warp
    loss_f = 0
    curr_warp = torch.zeros_like(hr_frms[:, 0, :, :, :])
    # forward propagation
    for i in range(0, t):
        curr = hr_frms[:, i, :, :, :]
        if i > 0:  # no warping required for the first timestep
            flow = bwd_flows[:, i - 1, :, :, :]
            curr_warp = flow_warp(curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_f += F.l1_loss(
                weight_maps[:, i, :, :, :] * (1 - bwd_occs[:, i-1, :, :, :]) * prev, 
                weight_maps[:, i, :, :, :] * (1 - bwd_occs[:, i-1, :, :, :]) * curr
            )
        prev = curr_warp
    loss = loss_b + loss_f    
    return loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 diffloss_weight=1.0, temploss_weight=1.0, freqloss_weight=1.0, 
                 num_frames=5, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, 
                 disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", flownet_config=None):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.num_frames = num_frames
        self.kl_weight = kl_weight
        # tempo loss
        self.diffloss_weight = diffloss_weight
        self.temploss_weight = temploss_weight
        # pixel loss
        self.pixel_weight = pixelloss_weight
        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # freq loss
        # self.focalfreq_loss = FocalFrequencyLoss(loss_weight=freqloss_weight, alpha=1.0)
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        # flownet
        if flownet_config:
            self.instantiate_flownet(flownet_config)  # load flownet path
        # discriminator
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def instantiate_flownet(self, config):
        model = instantiate_from_config(config)
        self.flownet = model
        # disable flownet training
        for param in self.flownet.parameters():
            param.requires_grad = False

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None, return_dic=False):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.mean(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss) / nll_loss.shape[0]
        
        # # focal freq loss
        # weighted_nll_loss += self.focalfreq_loss(inputs.contiguous(), reconstructions.contiguous())
        
        # diff loss
        diff_loss = l1_diff(inputs.contiguous(), reconstructions.contiguous(), t=self.num_frames)
        weighted_nll_loss += self.diffloss_weight * torch.mean(diff_loss) / diff_loss.shape[0]

        # temp loss
        temp_loss = swc_loss(inputs.contiguous(), reconstructions.contiguous(), t=self.num_frames, flownet=self.flownet, w=3)
        weighted_nll_loss += self.temploss_weight * torch.mean(temp_loss)

        if self.kl_weight>0:
            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    # assert not self.training
                    d_weight = torch.tensor(1.0) * self.discriminator_weight
            else:
                # d_weight = torch.tensor(0.0)
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if self.kl_weight>0:
                loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                       "{}/logvar".format(split): self.logvar.detach(),
                       "{}/kl_loss".format(split): kl_loss.detach().mean(),
                       "{}/nll_loss".format(split): nll_loss.detach().mean(),
                       "{}/rec_loss".format(split): rec_loss.detach().mean(),
                       "{}/d_weight".format(split): d_weight.detach(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/g_loss".format(split): g_loss.detach().mean(),
                       }
                if return_dic:
                    loss_dic = {}
                    loss_dic['total_loss'] = loss.clone().detach().mean()
                    loss_dic['logvar'] = self.logvar.detach()
                    loss_dic['kl_loss'] = kl_loss.detach().mean()
                    loss_dic['nll_loss'] = nll_loss.detach().mean()
                    loss_dic['rec_loss'] = rec_loss.detach().mean()
                    loss_dic['d_weight'] = d_weight.detach()
                    loss_dic['disc_factor'] = torch.tensor(disc_factor)
                    loss_dic['g_loss'] = g_loss.detach().mean()
            else:
                loss = weighted_nll_loss + d_weight * disc_factor * g_loss
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                       "{}/logvar".format(split): self.logvar.detach(),
                       "{}/nll_loss".format(split): nll_loss.detach().mean(),
                       "{}/rec_loss".format(split): rec_loss.detach().mean(),
                       "{}/d_weight".format(split): d_weight.detach(),
                       "{}/disc_factor".format(split): torch.tensor(disc_factor),
                       "{}/g_loss".format(split): g_loss.detach().mean(),
                       }
                if return_dic:
                    loss_dic = {}
                    loss_dic["{}/total_loss".format(split)] = loss.clone().detach().mean()
                    loss_dic["{}/logvar".format(split)] = self.logvar.detach()
                    loss_dic['nll_loss'.format(split)] = nll_loss.detach().mean()
                    loss_dic['rec_loss'.format(split)] = rec_loss.detach().mean()
                    loss_dic['d_weight'.format(split)] = d_weight.detach()
                    loss_dic['disc_factor'.format(split)] = torch.tensor(disc_factor)
                    loss_dic['g_loss'.format(split)] = g_loss.detach().mean()

            if return_dic:
                return loss, log, loss_dic
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            if return_dic:
                loss_dic = {}
                loss_dic["{}/disc_loss".format(split)] = d_loss.clone().detach().mean()
                loss_dic["{}/logits_real".format(split)] = logits_real.detach().mean()
                loss_dic["{}/logits_fake".format(split)] = logits_fake.detach().mean()
                return d_loss, log, loss_dic

            return d_loss, log
