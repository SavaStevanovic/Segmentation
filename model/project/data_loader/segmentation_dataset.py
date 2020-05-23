import torch.utils.data as data
import os 
from PIL import Image
import torch

class SegmentationDataLoader(data.Dataset):
    def __init__(self, image_path, label_path, transform, testing):
        super(SegmentationDataLoader, self).__init__()
        self.transform = transform
        labels_names = set([x.replace('_maska_g.jpg', '.jpg') for x in os.listdir(label_path)])
        images_with_labels = labels_names.intersection(os.listdir(image_path)) 
        self.data = [(os.path.join(image_path, x), os.path.join(label_path, x.replace('.jpg', '_maska_g.jpg'))) for x in images_with_labels]
        if testing:
            self.data = self.data[:10]

    def set_transform(transform):
        self.transform = transform

    def __getitem__(self, index):
            img_path, mask_path = self.data[index]
            data = Image.open(img_path)
            label = Image.open(mask_path).convert('L')
            if self.transform is not None:
                data, label = self.transform(data, label)
            return data, label

    def __len__(self):
        return len(self.data)