import cv2
import math
import time
import os
import os.path as osp
import numpy as np
import random
import torch

from copy import deepcopy
from pathlib import Path
from torch.utils import data as data

from basicsr.data.mmcv_transforms import Clip, UnsharpMasking, RescaleToZeroOne
from basicsr.data.mmcv_transforms import RandomBlur, RandomResize, RandomNoise, RandomJPEGCompression, RandomVideoCompression, DegradationsWithShuffle
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment, single_random_crop, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img, imwrite
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
class RealVSRRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
    """

    def __init__(self, opt):
        super(RealVSRRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        # the first degradation
        self.random_blur_1 = RandomBlur(
            params=opt['degradation_1']['random_blur']['params'],
            keys=opt['degradation_1']['random_blur']['keys']
        )
        self.random_resize_1 = RandomResize(
            params=opt['degradation_1']['random_resize']['params'],
            keys=opt['degradation_1']['random_resize']['keys']
        )
        self.random_noise_1 = RandomNoise(
            params=opt['degradation_1']['random_noise']['params'],
            keys=opt['degradation_1']['random_noise']['keys']
        )
        self.random_jpeg_1 = RandomJPEGCompression(
            params=opt['degradation_1']['random_jpeg']['params'],
            keys=opt['degradation_1']['random_jpeg']['keys']
        )
        self.random_mpeg_1 = RandomVideoCompression(
            params=opt['degradation_1']['random_mpeg']['params'],
            keys=opt['degradation_1']['random_mpeg']['keys']
        )

        # the second degradation
        self.random_blur_2 = RandomBlur(
            params=opt['degradation_2']['random_blur']['params'],
            keys=opt['degradation_2']['random_blur']['keys']
        )
        self.random_resize_2 = RandomResize(
            params=opt['degradation_2']['random_resize']['params'],
            keys=opt['degradation_2']['random_resize']['keys']
        )
        self.random_noise_2 = RandomNoise(
            params=opt['degradation_2']['random_noise']['params'],
            keys=opt['degradation_2']['random_noise']['keys']
        )
        self.random_jpeg_2 = RandomJPEGCompression(
            params=opt['degradation_2']['random_jpeg']['params'],
            keys=opt['degradation_2']['random_jpeg']['keys']
        )
        self.random_mpeg_2 = RandomVideoCompression(
            params=opt['degradation_2']['random_mpeg']['params'],
            keys=opt['degradation_2']['random_mpeg']['keys']
        )

        # final
        self.resize_final = RandomResize(
            params=opt['degradation_2']['resize_final']['params'],
            keys=opt['degradation_2']['resize_final']['keys']
        )
        self.blur_final = RandomBlur(
            params=opt['degradation_2']['blur_final']['params'],
            keys=opt['degradation_2']['blur_final']['keys']
        )

        # transforms
        self.usm = UnsharpMasking(
            kernel_size=opt['transforms']['usm']['kernel_size'],
            sigma=opt['transforms']['usm']['sigma'],
            weight=opt['transforms']['usm']['weight'],
            threshold=opt['transforms']['usm']['threshold'],
            keys=opt['transforms']['usm']['keys']
        )
        self.clip = Clip(keys=opt['transforms']['clip']['keys'])
        self.rescale = RescaleToZeroOne(keys=opt['transforms']['rescale']['keys'])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gts.append(img_gt)

        # randomly crop
        img_gts = single_random_crop(img_gts, gt_size, img_gt_path)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])
        img_lqs = deepcopy(img_gts)

        out_dict = {'lqs': img_lqs, 'gts': img_gts}

        out_dict = self.usm.transform(out_dict)

        ## the first degradation
        out_dict = self.random_blur_1(out_dict)
        out_dict = self.random_resize_1(out_dict)
        out_dict = self.random_noise_1(out_dict)
        out_dict = self.random_jpeg_1(out_dict)
        out_dict = self.random_mpeg_1(out_dict)

        ## the second degradation
        out_dict = self.random_blur_2(out_dict)
        out_dict = self.random_resize_2(out_dict)
        out_dict = self.random_noise_2(out_dict)
        out_dict = self.random_jpeg_2(out_dict)
        out_dict = self.random_mpeg_2(out_dict)

        ## final resize
        out_dict = self.resize_final(out_dict)
        out_dict = self.blur_final(out_dict)

        # post process
        out_dict = self.clip(out_dict)
        out_dict = self.rescale.transform(out_dict)

        # list-to-list
        for k in out_dict.keys():
            out_dict[k] = img2tensor(out_dict[k])

        # img_gts: (t, c, h, w)
        # key: str

        return out_dict

    def __len__(self):
        return len(self.keys)


# @DATASET_REGISTRY.register()
class RealESRGANRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
    """

    def __init__(self, opt):
        super(RealESRGANRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root = Path(opt['dataroot_gt'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts = single_random_crop(img_gts, gt_size, img_gt_path)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        # list-to-list
        img_gts = img2tensor(img_gts)

        # kernels
        kernel1s = []
        kernel2s = []
        sinc_kernels = []

        for _ in range(len(img_gts)):
            # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob']:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel = torch.FloatTensor(kernel)

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.opt['sinc_prob2']:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None)

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel2 = torch.FloatTensor(kernel2)

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.opt['final_sinc_prob']:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            kernel1s.append(kernel)
            kernel2s.append(kernel2)
            sinc_kernels.append(sinc_kernel)

        # img_gts: (t, c, h, w)
        # key: str
        return {'gts': img_gts, 'kernel1s': kernel1s, 'kernel2s': kernel2s, 'sinc_kernels': sinc_kernels, 'keys': key}

    def __len__(self):
        return len(self.keys)
