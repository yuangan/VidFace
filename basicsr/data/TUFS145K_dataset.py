import mmcv
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.utils import FileClient, get_root_logger
import pickle

def normalize_list(img_results):
    out = []
    for i in img_results:
        out.append((i - 0.5) * 2)
    return out

def readin_lmk():
    f = open('./landmarks/tufs145k_lmk_norm.pickle', 'rb')
    dict_lmk = pickle.load(f)
    return dict_lmk

class TUFS145KDataset(data.Dataset):
    """TUFS145K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_TUFS145K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        id00012/21Uxsk56VDQ 7 (16, 12, 3)
        id00012/2DLq_Kkc1r8 7 (16, 12, 3)
        id00012/73OrGYvy4ng 7 (16, 12, 3)

    Key examples: "id00012/21Uxsk56VDQ"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(TUFS145KDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(
            opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [
            i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])
        ]

        # temporal augmentation configs
        self.random_reverse = False
        logger = get_root_logger()
        logger.info(f'Random reverse is {self.random_reverse}.')
        print('-----THIS WILL ONLY RUN ONCE FOR EVERY DATALOADER-----')
        self.all_lmk = readin_lmk()

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        # shuffle neighbor list to expand train dataset        
        random.shuffle(self.neighbor_list)

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ frames
        img_lqs = []
        img_gts = []
        count = 0
        lmks = []
        tmp_lmk = np.zeros(10)
        flags = []
        for neighbor in self.neighbor_list:
            count += 1
            img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            tmp_gt = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
            img_gts.append(tmp_gt)
            
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
            img_lqs.append(img_lq)
            if f'{clip}/{seq}/im{neighbor}.png' in self.all_lmk.keys():
                lmk = self.all_lmk[f'{clip}/{seq}/im{neighbor}.png']
                lmks.append(lmk)
                flags.append(1)
            else:
                lmks.append(tmp_lmk)
                flags.append(0)
        lmks = torch.from_numpy(np.stack(lmks)).float() - 0.5
        flags = torch.from_numpy(np.stack(flags)).float()

        # augmentation - flip
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'],
                              self.opt['use_rot'])
        
        img_results = totensor(img_results)

        # normalize image
        img_results = normalize_list(img_results)

        # split augmented image
        img_lqs = torch.stack(img_results[0:-7], dim=0)
        img_gts = torch.stack(img_results[-7:], dim=0)

        return {'lq': img_lqs, 'key': key, 'gts': img_gts, 'lmks': lmks, 'flags': flags}

    def __len__(self):
        return len(self.keys)
