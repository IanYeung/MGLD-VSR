import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import cv2
from util_image import ImageSpliterTh
from pathlib import Path
from basicsr.archs.arch_util import resize_flow
from scripts.util_flow import forward_backward_consistency_check
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    return 2.*image - 1.


def read_image(im_path):
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32) / 255.0
    im = im[None].transpose(0, 3, 1, 2)
    im = (torch.from_numpy(im) - 0.5) / 0.5

    return im


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seqs-path",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload"
    )
    # parser.add_argument(
    #     "--init-img",
    #     type=str,
    #     nargs="?",
    #     help="path to the input image",
    #     default="inputs/user_upload"
    # )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="device",
        default="cuda"
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=5,
        help="number of frames to perform inference",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/stablevsr_025.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="checkpoints/vqgan_cfw_00011.ckpt",
        help="path to checkpoint of VQGAN model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--select_idx",
        type=int,
        default=0,
        help="selected sequence index",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpu for testing",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=32,
        help="tile overlap size (in latent)",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=4.0,
        help="upsample scale",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )
    parser.add_argument(
        "--vqgantile_stride",
        type=int,
        default=750,
        help="the stride for tile operation before VQGAN decoder (in pixel)",
    )
    parser.add_argument(
        "--vqgantile_size",
        type=int,
        default=960,
        help="the size for tile operation before VQGAN decoder (in pixel)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    print('>>>>>>>>>>color correction>>>>>>>>>>>')
    if opt.colorfix_type == 'adain':
        print('Use adain color correction')
    elif opt.colorfix_type == 'wavelet':
        print('Use wavelet color correction')
    else:
        print('No color correction')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # Testing
    select_idx = opt.select_idx
    num_gpu_test = opt.n_gpus

    # Model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    model.configs = config

    vqgan_config = OmegaConf.load("configs/video_autoencoder/video_autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                            linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000

    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []
    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
    model.num_timesteps = 1000
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()
    model = model.to(device)
    model.cond_stage_model.device = device

    # Data
    n_frames = opt.n_frames
    batch_size = opt.n_samples

    seq_name_list = sorted(os.listdir(opt.seqs_path))
    for seq_idx, seq_item in enumerate(seq_name_list):
        if seq_idx % num_gpu_test != select_idx:
           continue
        seq_path = os.path.join(opt.seqs_path, seq_item)
        temp_frame_buffer = []
        init_segment_list = []
        img_name_list = sorted(os.listdir(seq_path))
        # naive way to handle the boundary problem
        while len(img_name_list) % n_frames != 0:
            img_name_list.append(img_name_list[-1])
        for idx, item in enumerate(img_name_list):
            # img_name = item.split('/')[-1]
            cur_image = read_image(os.path.join(seq_path, item))
            size_min = min(cur_image.size(-1), cur_image.size(-2))
            upsample_scale = max(512 / size_min, opt.upscale)
            cur_image = F.interpolate(
                cur_image,
                size=(int(cur_image.size(-2) * upsample_scale),
                      int(cur_image.size(-1) * upsample_scale)),
                mode='bicubic',
            )
            temp_frame_buffer.append(cur_image)
            if idx % n_frames == n_frames - 1:
                # [1, c, h, w] -> [b, c, h, w]
                temp_frame = torch.cat(copy.deepcopy(temp_frame_buffer), dim=0)
                print('Segment shape: ', temp_frame.shape)
                init_segment_list.append(temp_frame)
                temp_frame_buffer.clear()

        # print(f"Found {len(img_name_list)} inputs.")

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        niqe_list = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(len(init_segment_list), desc="Sampling"):
                        init_image = init_segment_list[n]
                        im_lq_bs = init_image.clamp(-1.0, 1.0).to(device)

                        print('>>>>>>>>>>>>>>>>>>>>>>>')
                        print('seq: ', seq_item, 'seg: ', n, 'size: ', im_lq_bs.size())
                        ori_h, ori_w = im_lq_bs.shape[2:]

                        ref_patch=None
                        if not (ori_h % 32 == 0 and ori_w % 32 == 0):
                            flag_pad = True
                            pad_h = ((ori_h // 32) + 1) * 32 - ori_h
                            pad_w = ((ori_w // 32) + 1) * 32 - ori_w
                            im_lq_bs = F.pad(im_lq_bs, pad=(0, pad_w, 0, pad_h), mode='reflect')
                        else:
                            flag_pad = False
                        
                        im_lq_bs_0_1 = torch.clamp((im_lq_bs + 1.0) / 2.0, min=0.0, max=1.0)
                        _, _, im_h, im_w = im_lq_bs_0_1.shape
                        
                        # flow estimation
                        im_lq_bs_0_1 = F.interpolate(im_lq_bs_0_1, size=(im_h//4, im_w//4), mode='bicubic')
                        im_lq_bs_0_1 = rearrange(im_lq_bs_0_1, '(b t) c h w -> b t c h w', t=init_image.size(0))
                        flows = model.compute_flow(im_lq_bs_0_1)  # flows: [flows_bwd(forward_prop), flows_fwd(backward_prop)]
                        flows = [rearrange(flow, 'b t c h w -> (b t) c h w') for flow in flows]
                        flows = [resize_flow(flow, size_type='shape', sizes=(im_h//8, im_w//8)) for flow in flows]
                        flows = [rearrange(flow, '(b t) c h w -> b t c h w', t=init_image.size(0)-1) for flow in flows]

                        # occlusion mask estimation
                        fwd_occ_list, bwd_occ_list = list(), list()
                        for i in range(init_image.size(0)-1):
                            fwd_flow, bwd_flow = flows[1][:, i, :, :, :], flows[0][:, i, :, :, :]
                            fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)
                            fwd_occ_list.append(fwd_occ.unsqueeze_(1))
                            bwd_occ_list.append(bwd_occ.unsqueeze_(1))
                        fwd_occs = torch.stack(fwd_occ_list, dim=1)
                        fwd_occs = rearrange(fwd_occs, 'b t c h w -> (b t) c h w')
                        bwd_occs = torch.stack(bwd_occ_list, dim=1)
                        bwd_occs = rearrange(bwd_occs, 'b t c h w -> (b t) c h w')
                        # masks = [fwd_occ_list, bwd_occ_list]

                        flows = [rearrange(flow, 'b t c h w -> (b t) c h w') for flow in flows]

                        if im_lq_bs.shape[2] > opt.vqgantile_size or im_lq_bs.shape[3] > opt.vqgantile_size:

                            imlq_spliter = ImageSpliterTh(im_lq_bs, opt.vqgantile_size, opt.vqgantile_stride, sf=1)
                            flow_spliter_f = ImageSpliterTh(flows[0], opt.vqgantile_size // 8, opt.vqgantile_stride // 8, sf=1)
                            flow_spliter_b = ImageSpliterTh(flows[1], opt.vqgantile_size // 8, opt.vqgantile_stride // 8, sf=1)
                            fwd_occ_spliter = ImageSpliterTh(fwd_occs, opt.vqgantile_size // 8, opt.vqgantile_stride // 8, sf=1)
                            bwd_occ_spliter = ImageSpliterTh(bwd_occs, opt.vqgantile_size // 8, opt.vqgantile_stride // 8, sf=1)

                            for (im_lq_pch, index_infos), (flow_f, _), (flow_b, _), (fwd_occ, _), (bwd_occ, _) \
                                in zip(imlq_spliter, flow_spliter_f, flow_spliter_b, fwd_occ_spliter, bwd_occ_spliter):
                                seed_everything(opt.seed)
                                init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_pch))
                                text_init = ['']*opt.n_samples
                                semantic_c = model.cond_stage_model(text_init)
                                noise = torch.randn_like(init_latent)
                                # If you would like to start from the intermediate steps,
                                # you can add noise to LR to the specific steps.
                                t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
                                t = t.to(device).long()
                                x_T = model.q_sample_respace(x_start=init_latent,
                                                             t=t,
                                                             sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                                                             sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                                                             noise=noise)
                                # x_T = noise
                                # im_lq_pch_0_1 = torch.clamp((im_lq_pch + 1.0) / 2.0, min=0.0, max=1.0)
                                flow_f = rearrange(flow_f, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                                flow_b = rearrange(flow_b, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                                fwd_occ = rearrange(fwd_occ, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                                bwd_occ = rearrange(bwd_occ, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                                flows = (flow_f, flow_b)
                                masks = (fwd_occ, bwd_occ)
                                samples, _ = model.sample_canvas(cond=semantic_c,
                                                                 struct_cond=init_latent,
                                                                 guidance_scale=-10.0,
                                                                 lr_images=None,
                                                                 flows=flows,
                                                                 masks=masks,
                                                                 cond_flow=None,
                                                                 batch_size=im_lq_pch.size(0),
                                                                 timesteps=opt.ddpm_steps,
                                                                 time_replace=opt.ddpm_steps,
                                                                 x_T=x_T,
                                                                 return_intermediates=True,
                                                                 tile_size=64,
                                                                 tile_overlap=opt.tile_overlap,
                                                                 batch_size_sample=opt.n_samples)
                                _, enc_fea_lq = vq_model.encode(im_lq_pch)
                                x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                                # x_samples = model.decode_first_stage(samples)

                                if opt.colorfix_type == 'adain':
                                    x_samples = adaptive_instance_normalization(x_samples, im_lq_pch)
                                elif opt.colorfix_type == 'wavelet':
                                    x_samples = wavelet_reconstruction(x_samples, im_lq_pch)
                                imlq_spliter.update(x_samples, index_infos)
                            im_sr = imlq_spliter.gather()
                            im_sr = torch.clamp((im_sr+1.0)/2.0, min=0.0, max=1.0)

                        else:
                            init_latent = model.get_first_stage_encoding(model.encode_first_stage(im_lq_bs))
                            text_init = ['']*opt.n_samples
                            semantic_c = model.cond_stage_model(text_init)
                            noise = torch.randn_like(init_latent)
                            # If you would like to start from the intermediate steps,
                            # you can add noise to LR to the specific steps.
                            t = repeat(torch.tensor([999]), '1 -> b', b=im_lq_bs.size(0))
                            t = t.to(device).long()
                            x_T = model.q_sample_respace(x_start=init_latent,
                                                         t=t,
                                                         sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                                                         sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                                                         noise=noise)
                            # x_T = noise
                            flow_f = rearrange(flow_f, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                            flow_b = rearrange(flow_b, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                            fwd_occ = rearrange(fwd_occ, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                            bwd_occ = rearrange(bwd_occ, '(b t) c h w -> b t c h w', t=init_image.size(0)-1)
                            flows = (flow_f, flow_b)
                            masks = (fwd_occ, bwd_occ)
                            samples, _ = model.sample_canvas(cond=semantic_c,
                                                             struct_cond=init_latent,
                                                             guidance_scale=-10.0,
                                                             lr_images=None,
                                                             flows=flows,
                                                             masks=masks,
                                                             cond_flow=None,
                                                             batch_size=im_lq_bs.size(0),
                                                             timesteps=opt.ddpm_steps,
                                                             time_replace=opt.ddpm_steps,
                                                             x_T=x_T,
                                                             return_intermediates=True,
                                                             tile_size=64,
                                                             tile_overlap=opt.tile_overlap,
                                                             batch_size_sample=opt.n_samples)
                            _, enc_fea_lq = vq_model.encode(im_lq_bs)
                            x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                            # x_samples = model.decode_first_stage(samples)

                            if opt.colorfix_type == 'adain':
                                x_samples = adaptive_instance_normalization(x_samples, im_lq_bs)
                            elif opt.colorfix_type == 'wavelet':
                                x_samples = wavelet_reconstruction(x_samples, im_lq_bs)
                            im_sr = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if upsample_scale > opt.upscale:
                            im_sr = F.interpolate(
                                        im_sr,
                                        size=(int(im_lq_bs.size(-2)*opt.upscale/upsample_scale),
                                              int(im_lq_bs.size(-1)*opt.upscale/upsample_scale)),
                                        mode='bicubic',
                                        )
                            im_sr = torch.clamp(im_sr, min=0.0, max=1.0)

                        im_sr = im_sr.cpu().numpy().transpose(0, 2, 3, 1) * 255   # b x h x w x c

                        if flag_pad:
                            im_sr = im_sr[:, :ori_h, :ori_w, ]

                        os.makedirs(os.path.join(opt.outdir, seq_item), exist_ok=True)
                        for jj in range(im_lq_bs.shape[0]):
                            img_name = img_name_list.pop(0)
                            # img_name = str(Path(im_path_bs[jj]).name)
                            basename = os.path.splitext(os.path.basename(img_name))[0]
                            outpath = os.path.join(opt.outdir, seq_item, '{}.png'.format(basename))
                            Image.fromarray(im_sr[jj, ].astype(np.uint8)).save(outpath)
                        
                        # occdir = '/home/notebook/data/personal/S9049909/Results/StableVSRResults/occs'
                        # os.makedirs(os.path.join(occdir, seq_item), exist_ok=True)
                        # for ii in range(bwd_occs.size(0)):
                        #     occ = bwd_occs[ii, :, :, :].squeeze().detach().cpu().numpy()  # [H, W]
                        #     occ = 255. * occ.astype(np.uint8)
                        #     filename = os.path.join(occdir, seq_item, '{:08d}.png'.format(n*5+ii))
                        #     cv2.imwrite(filename, occ)
                    toc = time.time()


if __name__ == "__main__":
    main()
