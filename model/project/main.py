import torch
from model import blocks
from model import networks
from data_loader.dataset_provider import SegmentationDatasetProvider
from model_fitting.train import fit
import os


th_count = 12
dataset_name = 'custom_car'

net = networks.Unet(block = blocks.ConvBlock, inplanes = 64, in_dim=3, out_dim=2, depth=4, norm_layer=torch.nn.InstanceNorm2d)
data_provider = SegmentationDatasetProvider(net, batch_size=1, th_count=th_count)

fit(net, data_provider.trainloader, data_provider.validationloader, 
    dataset_name = dataset_name, 
    epochs = 1000, 
    lower_learning_period = 5)      