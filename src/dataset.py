import os
import random
import numpy as np
import pandas as pd

import cv2
import albumentations as A
import albumentations.pytorch as ATorch
from albumentations import ImageOnlyTransform
import pydicom

import torch
from torch.utils.data import Dataset
import jpeg4py

from . import config
from .common.logger import get_logger


class NormalizePerImage(ImageOnlyTransform):
    def __init__(self, max_pixel_value=255.0, always_apply=False, p=1.0):
        super(NormalizePerImage, self).__init__(always_apply, p)

    def apply(self, image, **params):
        """
        Parameters
        ----------
        image: np.ndarray of np.uint8 or np.float32, shape of [w, h, c]

        Returns
        -------
        normed: np.ndarray
        """
        normed = image.astype(np.float32)
        n_channels = image.shape[2]
        mean = np.mean(image.reshape(-1, n_channels), axis=0)
        std = np.std(image.reshape(-1, n_channels), axis=0)
        normed -= mean
        normed /= (std + 1e-8)
        return normed

    def get_trainsform_init_args_names(self):
        return 'max_pixel_value'


alb_trn_trnsfms = A.Compose([
    A.Resize(*config.IMG_SIZE),
    # A.CLAHE(p=1),
    A.Rotate(limit=10, p=1),
    # A.RandomSizedCrop((IMG_SIZE[0]-32, IMG_SIZE[0]-10), *INPUT_SIZE),
    A.RandomCrop(*config.INPUT_SIZE),
    # A.HueSaturationValue(val_shift_limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    A.GaussianBlur(blur_limit=7, p=0.5),
    NormalizePerImage(),
    # A.Normalize(
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225],
    # ),
    ATorch.transforms.ToTensor()
], p=1)


alb_val_trnsfms = A.Compose([
    A.Resize(*config.IMG_SIZE),
    # A.CLAHE(p=1),
    A.CenterCrop(*config.INPUT_SIZE),
    NormalizePerImage(),
    # A.Normalize(
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225],
    # ),
    ATorch.transforms.ToTensor()
], p=1)

alb_tst_trnsfms = A.Compose([
    A.Resize(*config.IMG_SIZE),
    # A.CLAHE(),
    A.CenterCrop(*config.INPUT_SIZE),
    NormalizePerImage(),
    # A.Normalize(
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225],
    # ),
    ATorch.transforms.ToTensor()
], p=1)


class BrainDataset(Dataset):
    target_name = ['epidural', 'intraparenchymal',
                   'intraventricular', 'subarachnoid', 'subdural', 'any']
    image_name = ''
    id_name = 'Image'

    def __init__(self, df, image_dir, transform, mode):
        self.df_org = df.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # Random Selection
        if mode == 'train':
            self.update()
            # self.df_selected = self.df_org
        elif mode in ['valid', 'predict']:
            self.df_selected = self.df_org
        else:
            raise ValueError('Unexpected mode: %s' % mode)

    def __len__(self):
        return self.df_selected.shape[0]

    def __getitem__(self, idx):
        image_name = self._get_image_name(self.df_selected, idx)
        try:
            image = self._load_image(image_name)
        except Exception as e:
            raise ValueError('Could not load image: %s' % image_name) from e

        if self.mode in ['train', 'valid']:
            label = self.df_selected.iloc[idx][self.target_name]
        elif self.mode == 'predict':
            label = -1
        return image, torch.tensor(label)

    def _get_image_name(self, df, idx):
        """
        get image name

        Returns
        -------
        file_path: str
            image file path
        """
        rcd = df.iloc[idx]
        image_id = rcd['Image']

        if config.USE_DCM:
            file_name = '%s.dcm' % image_id
        else:
            file_name = '%s.jpg' % image_id
        file_path = os.path.join(self.image_dir, file_name)
        return file_path

    def _load_image(self, image_path):
        """
        Parameters
        ----------
        image_path: str
            image file path
        """
        # image = cv2.imread(image_path)
        if config.USE_DCM:
            image = read_dicom(image_path)
        else:
            image = jpeg4py.JPEG(image_path).decode()

        if image is None:
            raise ValueError('Not found image: %s' % image_path)

        augmented = self.transform(image=image)
        image = augmented['image']
        return image

    def update(self):
        if self.mode != 'train':
            raise ValueError('CellerDataset is not train mode.')
        # self.df_selected = self.random_selection(
            # config.N_CLASSES, config.N_SAMPLES)
        self.df_selected = self.random_selection2(config.N_SAMPLES)

    def random_selection(self, n_classes, n_samples):
        """select images as each category is same rate"""
        g = self.df_org.groupby('any')[self.id_name]
        selected = [np.random.choice(g.get_group(i).tolist(), n_samples, replace=False)
                    for i in range(n_classes)]
        selected = np.concatenate(selected, axis=0)

        df_new = pd.DataFrame({self.id_name: selected})
        df_new = df_new.merge(self.df_org, on=self.id_name, how='left')
        # shuffle
        df_new = df_new.sample(frac=1, random_state=2019)
        get_logger().info('num of selected_images: %d' % len(df_new))

        return df_new

    def random_selection2(self, n_samples):
        df_new = self.df_org.sample(n=n_samples, random_state=2019)
        get_logger().info('num of selected_images: %d' % len(df_new))

        return df_new


class BrainTTADataset(BrainDataset):
    def __init__(self, df, image_dir, transform, mode, n_tta):
        super(BrainTTADataset, self).__init__(df, image_dir, transform, mode)

        self.n_tta = n_tta

    def __getitem__(self, idx):
        """
        Return augmented images and labe(-1)
        """
        image_path = self._get_image_name(self.df_selected, idx)
        try:
            image = jpeg4py.JPEG(image_path).decode()
            images = []
            for _ in range(self.n_tta):
                image_new = image.copy()
                augmented = self.transform(image=image_new)
                image_new = augmented['image']
                images.append(image_new)

        except Exception as e:
            raise ValueError('Could not load image: %s' % image_path) from e

        label = -1

        return images, torch.tensor(label)


WINDOW_LIST = [
    (40, 80),
    (80, 200),
    (40, 380)
]

WINDOW_MIN_MAX_LIST = [
    (0, 80),
    (-20, 180),
    (-150, 230)
]


def read_dicom(image_path):
    data = pydicom.read_file(image_path)
    image = data.pixel_array
    window_center, window_width, intercept, slope = get_windowing(data)
    image = (image * slope + intercept)

    '''
    windowed_images = []
    for window_center, window_width in WINDOW_LIST:
        image_windowed = window_image_no_slope(
            image, window_center, window_width)
        windowed_images.append(image_windowed)
    '''
    windowed_images = [window_image_light(
        image, win_center, win_width) for win_center, win_width in WINDOW_MIN_MAX_LIST]
    windowed_images = np.stack(windowed_images, axis=2)

    windowed_images = windowed_images.astype(np.float32)
    return windowed_images


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def window_image(img, window_center, window_width, intercept, slope):
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    img = (img - img_min) / (img_max - img_min)

    return img


def window_image_light(img_org, img_min, img_max):
    img = img_org.copy()
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    img = (img - img_min) / (img_max - img_min)

    return img
