import torch
from data_loader.segmentation_dataset import SegmentationDataLoader
from data_loader import augmentation
import multiprocessing as mu
import os

class SegmentationDatasetProvider():
    def __init__(self, net,  batch_size=1, train_transforms=None, val_transforms=None, th_count=mu.cpu_count()):
        if train_transforms is None:
            train_transforms = augmentation.PairCompose([
                                            augmentation.ResizeDownToSizeTransform(512),
                                            augmentation.RandomResizeTransform(),
                                            augmentation.RandomHorizontalFlipTransform(),
                                            augmentation.RandomCropTransform((256, 256)),
                                            # augmentation.RandomNoiseTransform(),
                                            augmentation.RandomColorJitterTransform(),
                                            # augmentation.RandomBlurTransform(),
                                            augmentation.OutputTransform()])
        if val_transforms is None:
            val_transforms = augmentation.PairCompose([
                                            augmentation.ResizeDownToSizeTransform(512),
                                            augmentation.PaddTransform(pad_size=2**net.depth), 
                                            augmentation.OutputTransform()])

        self.trainset      = SegmentationDataLoader(image_path = '/Data/dataset_10224_raw/raw_split/train/',      label_path = '/Data/dataset_10224_glass_full/glass_full', transform = train_transforms, testing = th_count==1)
        self.validationset = SegmentationDataLoader(image_path = '/Data/dataset_10224_raw/raw_split/validation/', label_path = '/Data/dataset_10224_glass_full/glass_full', transform = val_transforms,   testing = th_count==1)
        self.testset       = SegmentationDataLoader(image_path = '/Data/dataset_10224_raw/raw_split/test/',       label_path = '/Data/dataset_10224_glass_full/glass_full', transform = val_transforms,   testing = th_count==1)

        self.trainloader      = torch.utils.data.DataLoader(self.trainset,      batch_size=batch_size, shuffle=True , num_workers=th_count, pin_memory=True)
        self.validationloader = torch.utils.data.DataLoader(self.validationset, batch_size=1,          shuffle=False, num_workers=th_count, pin_memory=True)
        self.testloader       = torch.utils.data.DataLoader(self.testset,       batch_size=1,          shuffle=False, num_workers=th_count, pin_memory=True)