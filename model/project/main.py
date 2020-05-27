import torch
from model import blocks
from model import networks
from data_loader.dataset_provider import SegmentationDatasetProvider
from model_fitting.train import fit, test
import os


th_count = 12
dataset_name = 'custom_car'

net_backbone = networks.ResNetBackbone(block = blocks.BasicBlock, block_counts = [3, 4, 6], inplanes=64)
net = networks.DeepLabV3Plus(net_backbone, 2)
data_provider = SegmentationDatasetProvider(net, batch_size=4, th_count=th_count)

fit(net, data_provider.trainloader, data_provider.validationloader, 
    dataset_name = dataset_name, 
    epochs = 1000, 
    lower_learning_period = 5)      

test(net, data_provider.testloader, dataset_name = dataset_name)