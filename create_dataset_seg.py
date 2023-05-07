import torch
from torch.utils import data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
class Mydataset(data.Dataset):
    def __init__(self, img_paths, masks, transform):
        self.imgs = img_paths
        self.masks = masks
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = self.masks[index]
        img = self.transforms(image=img, mask=mask)
        return img['image'], (img['mask']/255).long()#, label

    def __len__(self):
        return len(self.imgs)


class Mydataset_class(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        labels = int(label)
        data = self.transforms(image=img)

        return data['image'], labels

    def __len__(self):
        return len(self.imgs)





class Mydataset_dila(data.Dataset):
    def __init__(self, img_paths, masks,transform):
        self.imgs = img_paths
        self.masks = masks
        # self.masks_dila = masks_dila
        self.transforms = transform

    def __getitem__(self, index):
        img0 = self.imgs[index]
        mask = self.masks[index]
        # mask_dila = self.masks_dila[index]
        img = self.transforms(image=img0, mask=mask,mask2=mask)
        img_peng = img['mask'].numpy()
        kernel = np.ones((15,15),np.uint8)
        peng2 = cv2.dilate(img_peng,kernel)
        mask_dila = torch.from_numpy(peng2)
        return img['image'], (img['mask']/255).long(),(mask_dila/255).long()#(img['mask_d']/255).long()#, label

    def __len__(self):
        return len(self.imgs)


class Mydataset_test(data.Dataset):
    def __init__(self, image_name,img_paths, masks, transform):
        self.img_name = image_name
        self.imgs = img_paths
        self.masks = masks
        self.transforms = transform

    def __getitem__(self, index):
        name = self.img_name[index]
        img = self.imgs[index]
        mask = self.masks[index]
        img = self.transforms(image=img, mask=mask)
        return name,img['image'], (img['mask']/255).long()#, label

    def __len__(self):
        return len(self.imgs)

class Mydataset_test_consistant_loss(data.Dataset):
    def __init__(self, image_name,img_paths,  transform):
        self.img_name = image_name
        self.imgs = img_paths
        self.transforms = transform

    def __getitem__(self, index):
        name = self.img_name[index]
        img = self.imgs[index]
        img = self.transforms(image=img)
        return name,img['image']#, label

    def __len__(self):
        return len(self.imgs)


class Mydataset_test_consistant_loss_with_mask(data.Dataset):
    def __init__(self, image_name,img_paths,mask_paths,  transform):
        self.img_name = image_name
        self.imgs = img_paths
        self.masks = mask_paths
        self.transforms = transform

    def __getitem__(self, index):
        name = self.img_name[index]
        img = self.imgs[index]
        mask = self.masks[index]
        img = self.transforms(image=img,mask=mask)
        return name,img['image'],(img['mask']/255).long()#, label

    def __len__(self):
        return len(self.imgs)

class Mydataset_test_no_read(data.Dataset):
    def __init__(self, image_names,mask_names,  transform,res):
        self.img_name = image_names
        self.mask_name = mask_names
        self.transforms = transform
        self.resize = res
    def __getitem__(self, index):
        name = self.img_name[index].split('/')[-1]
        img_name = self.img_name[index]
        mask_name = self.mask_name[index]
        img = cv2.resize(cv2.imread(img_name), (self.resize,self.resize))[:,:,::-1]
        mask = cv2.resize(cv2.imread(mask_name), (self.resize,self.resize))[:,:,0]
        # img = self.imgs[index]
        # mask = self.masks[index]
        img = self.transforms(image=img, mask=mask)
        return name,img['image'], (img['mask']/255).long()#, label

    def __len__(self):
        return len(self.img_name)


# def for_train_transform(size):
#     aug_size=int(size/10)
#     train_transform = A.Compose([
#     A.RandomRotate90(),
#     A.Flip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
#     A.RandomBrightnessContrast(
#         brightness_limit=0.5,
#         contrast_limit=0.1,
#         p=0.5
#     ),
#     A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=100, val_shift_limit=80),
#     A.OneOf([
#         A.CoarseDropout(max_holes=100,max_height=aug_size,max_width=aug_size,fill_value=[239, 234, 238]),
#         A.GaussNoise()
#     ]),
#     A.OneOf([
#         A.ElasticTransform(),
#         A.GridDistortion(),
#         A.OpticalDistortion(distort_limit=0.5,shift_limit=0)
#     ]),
#     A.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std= [0.229, 0.224, 0.225],
#         max_pixel_value=255.0,
#         p=1.0
#     ),
#     ToTensorV2()], p=1.)
#     return train_transform
class Mydataset_class_seg(data.Dataset):
    def __init__(self, img_paths, masks, labels,transform):
        self.imgs = img_paths
        self.masks = masks
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img0 = self.imgs[index]
        mask = self.masks[index]
        label = self.labels[index]
        labels = int(label)
        img = self.transforms(image=img0, mask=mask)

        return img['image'], (img['mask']/255).long(),labels#, label

    def __len__(self):
        return len(self.imgs)





def for_train_transform():
    # aug_size=int(size/10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.1,
            p=0.5
        ),
        A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=100, val_shift_limit=80),
        # A.OneOf([
        #     A.CoarseDropout(max_holes=100,max_height=aug_size,max_width=aug_size,fill_value=[239, 234, 238]),
        #     A.GaussNoise()
        # ]),
        A.GaussNoise(),
        A.OneOf([
             A.ElasticTransform(),
             A.GridDistortion(),
             A.OpticalDistortion(distort_limit=0.5,shift_limit=0)
         ]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform

def for_train_transform1():
    # aug_size=int(size/10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.5,
        #     contrast_limit=0.1,
        #     p=0.5
        # ),
        # A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=100, val_shift_limit=80),
        # A.OneOf([
        #     A.CoarseDropout(max_holes=100,max_height=aug_size,max_width=aug_size,fill_value=[239, 234, 238]),
        #     A.GaussNoise()
        # ]),
        # A.GaussNoise(),
        # A.OneOf([
        #     A.ElasticTransform(),
        #     A.GridDistortion(),
        #     A.OpticalDistortion(distort_limit=0.5,shift_limit=0)
        # ]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform


test_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)

import os
def get_image_paths(dataroot):

    paths = []
    if dataroot is not None:
        paths_img = os.listdir(dataroot)
        for _ in sorted(paths_img):
            path = os.path.join(dataroot, _)
            paths.append(path)
    return paths
class Mydataset_for_pre(data.Dataset):
    def __init__(self, img_paths,  resize,transform = test_transform):
        self.imgs = get_image_paths(img_paths)
        self.transforms = transform
        self.resize = resize
    def __getitem__(self, index):
        img_path = self.imgs[index]

        img = cv2.resize(cv2.imread(img_path), (self.resize,self.resize))[:,:,::-1]
        img = self.transforms(image=img)

        return img['image'],#, label

    def __len__(self):
        return len(self.imgs)