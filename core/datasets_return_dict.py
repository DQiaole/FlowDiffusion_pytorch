# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
import math
import random
from glob import glob
import os.path as osp

from .utils import frame_utils
from .utils.augmentor import FlowAugmentor, SparseFlowAugmentor


from .augmentations.augmentations import get_augmentation_fn as it_get_augmentation_fn
from .augmentations.aug_params import get_params as it_get_params


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, it_aug=False, resize_for_test=False, kitti_format=False, n_sample=None):
        self.augmentor = None
        self.sparse = sparse
        self.it_aug = it_aug
        self.resize_for_test = resize_for_test
        self.kitti_format = kitti_format
        self.n_sample_per_scene = n_sample
        if aug_params is not None:
            if not it_aug and 'add_gaussian_noise' in aug_params:
                aug_params.pop('add_gaussian_noise')
            if it_aug:
                params = it_get_params('pwc')
                if not aug_params['add_gaussian_noise']:
                    params.noise_std_range = 0.0
                    print('params.noise_std_range:', params.noise_std_range)
                self.augmentor = it_get_augmentation_fn(params)
            elif sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def inpaint_flow(self, flow, valid):
        flow[:, :, 0] = cv2.inpaint(flow[:, :, 0], 1 - valid.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        flow[:, :, 1] = cv2.inpaint(flow[:, :, 1], 1 - valid.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        return flow

    def read_data_for_training(self, index):
        valid = None
        if self.sparse or self.kitti_format:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            if self.kitti_format:
                valid = None
            if self.augmentor is None and self.sparse:
                flow = self.inpaint_flow(flow, valid)
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        return img1, img2, flow, valid

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        while True:
            img1, img2, flow, valid = self.read_data_for_training(index)

            if self.augmentor is not None:
                if self.resize_for_test:
                    if self.sparse:
                        scale_x = self.augmentor.crop_size[1] / flow.shape[1]
                        scale_y = self.augmentor.crop_size[0] / flow.shape[0]
                        img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                        img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                        flow, valid = self.augmentor.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
                        flow = self.inpaint_flow(flow, valid)
                    else:
                        raise NotImplementedError('Resize for dense flow')
                    break
                else:
                    try:
                        if self.it_aug:
                            element = {'inputs': [img1.astype(np.float32) / 127.5 - 1, img2.astype(np.float32) / 127.5 - 1],
                                       'label': flow}
                            element = self.augmentor(element)
                            img1, img2 = np.array(element['inputs'][0]), np.array(element['inputs'][1])
                            flow = np.array(element['label'])
                            img1 = (img1+1)*127.5
                            img2 = (img2+1)*127.5
                        elif self.sparse:
                            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                            flow = self.inpaint_flow(flow, valid)
                        else:
                            img1, img2, flow = self.augmentor(img1, img2, flow)
                        break
                    except:
                        print(self.image_list[index][0], self.image_list[index][1])
                        index = random.randint(0, len(self.image_list) - 1)
            else:
                break

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        batch = {
            'image0': img1,
            'image1': img2,
            'target': flow,
            'valid': valid.float(),
            'dataset_name': 'FlowDataset',
            'pair_names': (self.image_list[index][0], self.image_list[index][1]),
        }

        return batch

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class AutoFlow(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/AutoFlow', it_aug=True, n_sample=None):
        super(AutoFlow, self).__init__(aug_params, it_aug=it_aug, n_sample=n_sample)

        batches = sorted(
            glob(osp.join(root, 'static_40k_png_1_of_4/*')) + glob(osp.join(root, 'static_40k_png_2_of_4/*'))
            + glob(osp.join(root, 'static_40k_png_3_of_4/*')) + glob(osp.join(root, 'static_40k_png_4_of_4/*')))

        for i in range(len(batches)):
            batchid = batches[i]
            self.flow_list += [osp.join(batchid, 'forward.flo')]
            self.image_list += [[osp.join(batchid, 'im0.png'), osp.join(batchid, 'im1.png')]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass', split='training', n_sample=None):
        super(FlyingThings3D, self).__init__(aug_params, n_sample=n_sample)

        if split == 'training':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]
        elif split == 'validation':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TEST/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TEST/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]

                valid_list = np.loadtxt('things_val_test_set.txt', dtype=np.int32)
                self.image_list = [self.image_list[ind] for ind, sel in enumerate(valid_list) if sel]
                self.flow_list = [self.flow_list[ind] for ind, sel in enumerate(valid_list) if sel]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI', resize_for_test=False):
        super(KITTI, self).__init__(aug_params, sparse=True, resize_for_test=resize_for_test)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]

            seq_ix += 1
