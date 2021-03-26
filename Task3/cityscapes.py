import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import imutils
import cv2
import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', demo=False, *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')

        self.rootpth = rootpth
        self.cropsize = cropsize

        self.mode = mode
        self.demo = demo

        self.ignore_lb = 255

        with open('./cityscapes_info.json', 'r') as fr:
            self.labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in self.labels_info}
        self.class_map = {el['trainId']: el['id'] for el in self.labels_info}
        self.class_infos = {el['id']: el for el in self.labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomSelect([Resize(cropsize), RandomCrop(cropsize)])
            ])

        self.num_classes = set()
        for k, v in self.lb_map.items():
            if v < 0 or v == self.ignore_lb: continue
            
            self.num_classes.add(v)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        fn = self.imnames[idx]

        impth = self.imgs[fn]
        lbpth = self.labels[fn]

        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        if self.mode == 'train' and not self.demo:
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        else:
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_test(im_lb)
            img, label = im_lb['im'], im_lb['lb']

        img_t = self.to_tensor(img)
        label = np.array(label).astype(dtype=np.long)
        label = self.convert_labels(label)
        label = torch.from_numpy(label)

        if self.demo:
            if isinstance(img, np.ndarray) is False: img = np.array(img)
            return img, img_t, label

        return img_t, label

    def trans_test(self, im_lb):
        img, label = np.array(im_lb['im']), np.array(im_lb['lb'])

        W, H = self.cropsize

        dim = dict()
        if H < W:
            dim['width'] = W
        else:
            dim['height'] = H

        im_lb['im'] = imutils.resize(img, **dim)
        im_lb['lb'] = imutils.resize(label, inter=cv2.INTER_NEAREST, **dim)

        return im_lb

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v

        return label

    @property
    def n_classes(self):
        return len(self.num_classes)

    def get_class_info(self, train_id):
        return self.class_infos[self.class_map[train_id]]

    def vis_label(self, label):
        h, w = label.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(1, self.n_classes):
            class_info = self.get_class_info(class_id)

            mask[np.where(label == class_id)] = class_info['color']

        return mask

    def add_augmented_data(self):
        impth_root = os.path.join(self.rootpth, 'imgAug')
        lbpth_root = os.path.join(self.rootpth, 'gtAug')

        for root, _, filenames in os.walk(impth_root):
            for file_name in filenames:
                _id = file_name.replace('_leftImg8bit.png', '')

                name = f'AUG_DATA_{_id}'

                im_path = os.path.join(impth_root, file_name)
                lb_path = os.path.join(lbpth_root, f'{_id}_gtFine_labelIds.png')

                self.imnames.append(name)

                self.imgs[name] = im_path
                self.labels[name] = lb_path

        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())
        self.len = len(self.imnames)

    def shuffle(self):
        random.shuffle(self.imnames)


def main():
    cropsize = [384, 384]

    ds = CityScapes('/media/ssd/christen-rnd/Datasets/CItyscapes', cropsize=cropsize, mode='val', demo=True)
    n_classes = ds.n_classes

    dl = DataLoader(ds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True)

    for images, im, lb in dl:
        images = images.numpy()
        lb = lb.numpy()

        for image, label in zip(images, lb):
            label = ds.vis_label(label)

            cv2.imshow('image', image)
            cv2.imshow('label', label)

            if ord('q') == cv2.waitKey(0):
                exit()
        # exit()

    #     print(torch.unique(label))
    #     print(img.shape, label.shape)


if __name__ == "__main__":
    main()
