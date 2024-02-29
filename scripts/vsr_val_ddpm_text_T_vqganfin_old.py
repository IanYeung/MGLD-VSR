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
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seqs-path",
        type=str,
        nargs="?",
        help="path to the input image",
        default="inputs/user_upload"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/user_upload",
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=200,
        help="number of ddpm sampling steps",
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
        default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vqgan_ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
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
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--dec_w",
        type=float,
        default=0.5,
        help="weight for combining VQGAN and Diffusion",
    )
    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="nofix",
        help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
    )

    opt = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    vqgan_config = OmegaConf.load("configs/video_autoencoder/video_autoencoder_kl_64x64x4_resi.yaml")
    # vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = opt.dec_w

    seed_everything(opt.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)

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

    os.makedirs(opt.outdir, exist_ok=True)

    n_frames = opt.n_frames
    batch_size = opt.n_samples
    
    # inference time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    seq_name_list = sorted(os.listdir(opt.seqs_path))
    for seq_idx, seq_item in enumerate(seq_name_list):
        if seq_idx % num_gpu_test != select_idx:
            continue
        seq_path = os.path.join(opt.seqs_path, seq_item)
        temp_frame_buffer = []
        init_segment_list = []
        img_name_list = sorted(os.listdir(seq_path))

        for idx, item in enumerate(img_name_list):
            cur_image = load_img(os.path.join(opt.seqs_path, seq_item, item)).to(device)
            cur_image = transform(cur_image)
            cur_image = cur_image.clamp(-1, 1)
            temp_frame_buffer.append(cur_image)
            if idx % n_frames == n_frames - 1:
                # [1, c, h, w] -> [b, c, h, w]
                temp_frame = torch.cat(copy.deepcopy(temp_frame_buffer), dim=0)
                print('Segment shape: ', temp_frame.shape)
                init_segment_list.append(temp_frame)
                temp_frame_buffer.clear()
        # init_segment_list = torch.cat(init_segment_list, dim=0)
        # niters = math.ceil(init_segment_list.size(0) / batch_size)
        # init_segment_list = init_segment_list.chunk(niters)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        niqe_list = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    starter.record()
                    # tic = time.time()

                    all_samples = list()
                    for n in trange(len(init_segment_list), desc="Sampling"):
                        init_image = init_segment_list[n]
                        init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
                        init_latent = model.get_first_stage_encoding(init_latent_generator)
                        text_init = ['']*init_image.size(0)
                        semantic_c = model.cond_stage_model(text_init)

                        noise = torch.randn_like(init_latent)
                        # If you would like to start from the intermediate steps,
                        # you can add noise to LR to the specific steps.
                        t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                        t = t.to(device).long()
                        x_T = model.q_sample_respace(x_start=init_latent,
                                                     t=t,
                                                     sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                                                     sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                                                     noise=noise)
                        # x_T = None
                        init_image_0_1 = torch.clamp((init_image + 1.0) / 2.0, min=0.0, max=1.0).unsqueeze(0)
                        # init_image_0_1 = F.interpolate(init_image_0_1, scale_factor=0.125, mode='bicubic').unsqueeze(0)
                        flows = model.compute_flow(init_image_0_1)
                        flows = [rearrange(flow, 'b t c h w -> (b t) c h w') for flow in flows]
                        flows = [resize_flow(flow, size_type='ratio', sizes=(0.125, 0.125)) for flow in flows]
                        flows = [rearrange(flow, '(b t) c h w -> b t c h w', t=init_image.size(0)-1) for flow in flows]

                        # occlusion mask estimation
                        fwd_occ_list, bwd_occ_list = list(), list()
                        for i in range(init_image.size(0)-1):
                            fwd_flow, bwd_flow = flows[0][:, i, :, :, :], flows[1][:, i, :, :, :]
                            fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)
                            fwd_occ_list.append(fwd_occ.unsqueeze_(1))
                            bwd_occ_list.append(bwd_occ.unsqueeze_(1))
                        fwd_occs = torch.stack(fwd_occ_list, dim=1)
                        bwd_occs = torch.stack(bwd_occ_list, dim=1)
                        # masks = [fwd_occ_list, bwd_occ_list]
                        masks = (fwd_occs, bwd_occs)

                        samples, _ = model.sample(cond=semantic_c,
                                                  struct_cond=init_latent,
                                                  guidance_scale=-10.0,
                                                  lr_images=None,
                                                  flows=flows,
                                                  masks=masks,
                                                  cond_flow=None,
                                                  batch_size=1,
                                                  timesteps=opt.ddpm_steps,
                                                  time_replace=opt.ddpm_steps,
                                                  x_T=x_T,
                                                  return_intermediates=True)
                        x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                        if opt.colorfix_type == 'adain':
                            x_samples = adaptive_instance_normalization(x_samples, init_image)
                        elif opt.colorfix_type == 'wavelet':
                            x_samples = wavelet_reconstruction(x_samples, init_image)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        os.makedirs(os.path.join(opt.outdir, seq_item), exist_ok=True)
                        for i in range(init_image.size(0)):
                            img_name = img_name_list.pop(0)
                            basename = os.path.splitext(os.path.basename(img_name))[0]
                            x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                            img_out_path = os.path.join(opt.outdir, seq_item, '{}.png'.format(basename))
                            Image.fromarray(x_sample.astype(np.uint8)).save(img_out_path)

                    ender.record()
                    toc = time.time()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    print('Processing time: {:.4f} s'.format(curr_time / 1000))
                    # elapse_time = toc - tic
                    # print('Processing time: {:.4f} s'.format(elapse_time))


if __name__ == "__main__":
    main()
