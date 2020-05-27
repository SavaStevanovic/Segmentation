import random                     
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
from skimage import util
import torch 


class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomCropTransform(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, label):
        padding_x = max(self.crop_size[0]-image.size[0],0)
        padding_y = max(self.crop_size[1]-image.size[1],0)
        image = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        x = random.randint(self.crop_size[0]//2, image.size[0]-self.crop_size[0]//2)
        y = random.randint(self.crop_size[1]//2, image.size[1]-self.crop_size[1]//2)
        image = image.crop((x-self.crop_size[0]//2, y-self.crop_size[1]//2, x+self.crop_size[0]//2, y+self.crop_size[1]//2))
        label = label.crop((x-self.crop_size[0]//2, y-self.crop_size[1]//2, x+self.crop_size[0]//2, y+self.crop_size[1]//2))

        return image, label

class RandomHorizontalFlipTransform(object):
    def __call__(self, image, label):
        p = random.random()
        if p>0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)
        return image, label

class ResizeDownToSizeTransform(object):
    def __init__(self, hight):
        self.hight = hight

    def __call__(self, image, label):
        if image.size[-1]>self.hight:
            image = transforms.functional.resize(image, self.hight, Image.ANTIALIAS)
            label = transforms.functional.resize(label, self.hight, Image.ANTIALIAS)
        return image, label

class RandomResizeTransform(object):
    def __call__(self, image, label):
        p = random.random()/2+0.5
        new_size = [np.round(s*p).astype(np.int32) for s in list(image.size)[::-1]]
        image = transforms.functional.resize(image, new_size, Image.ANTIALIAS)
        label = transforms.functional.resize(label, new_size, Image.ANTIALIAS)
        return image, label

class RandomBlurTransform(object):
    def __init__(self):
        self.blur = ImageFilter.GaussianBlur(2)

    def __call__(self, image, label):
        p = random.random()
        if p>0.75:
            image = image.filter(self.blur)
        return image, label

class RandomColorJitterTransform(object):
    def __init__(self):
        self.jitt = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )

    def __call__(self, image, label):
        p = random.random()
        if p>0.75:
            image = self.jitt(image)
        return image, label

class RandomNoiseTransform(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        p = random.random()
        if p>0.75:
            image = np.array(image)
            image = util.random_noise(image, mode='gaussian', seed=None, clip=True)*255
            image = Image.fromarray(image.astype(np.uint8))
        return image, label

class PaddTransform(object):
    def __init__(self, pad_size = 32):
        self.pad_size = pad_size

    def __call__(self, image, label):
        padding_x = self.pad_size-image.size[0]%self.pad_size
        padding_x = (padding_x!=self.pad_size) * padding_x
        padding_y = self.pad_size-image.size[1]%self.pad_size
        padding_y = (padding_y!=self.pad_size) * padding_y
        image_padded = transforms.functional.pad(image, (0, 0, padding_x, padding_y))
        label_padded = transforms.functional.pad(label, (0, 0, padding_x, padding_y), fill=255)
        return image_padded, label_padded

class OutputTransform(object):
    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def __call__(self, image, label):
        image = self.toTensor(image)
        label = self.toTensor(label)
        label = (label>0.5).long().squeeze(0)
        return image, label