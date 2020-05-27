import torch
import os
from model_fitting.metrics import metrics
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from visualization import images_display
from model_fitting.configuration import TrainingConfiguration
import json
from functools import reduce
from torchsummary import summary
from sklearn.metrics import f1_score
from model_fitting.losses import SegmentationLoss
import numpy as np

def fit_epoch(net, dataloader, lr_rate, epoch=1):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    criterion = SegmentationLoss()
    losses = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    f1_metric = 0.0
    images = []
    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        optimizer.zero_grad()
        outputs = net(image.cuda())
        loss, focal_loss, dice_loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        losses += loss.item()
        total_focal_loss += focal_loss
        total_dice_loss += dice_loss
        outputs = outputs.softmax(1).detach().cpu().numpy()
        output_threshold = (outputs[:, 1] > 0.5).astype('float')
        f1_metric += f1_score(output_threshold.flatten(), labels.flatten(), average='weighted')

        if i>=len(dataloader)-5:
            images.append(( image.numpy(), 
                            outputs[:, 1], 
                            output_threshold,
                            labels.numpy()))
        
    data_len = len(dataloader)
    return losses/data_len, f1_metric/data_len, total_focal_loss/data_len, total_dice_loss/data_len, images

def fit(net, trainloader, validationloader, dataset_name, epochs=1000, lower_learning_period=10):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, '{}_checkpoints.pth'.format(dataset_name))
    checkpoint_conf_path = os.path.join(chp_dir, '{}_configuration.json'.format(dataset_name))
    train_config = TrainingConfiguration()
    if os.path.exists(chp_dir):
        net = torch.load(checkpoint_name_path)
        train_config.load(checkpoint_conf_path)
    net.cuda()
    summary(net, (3, 512, 512))
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
    for epoch in range(train_config.epoch, epochs):
        loss, f1_score, focal_loss, dice_loss, samples = fit_epoch(net, trainloader, train_config.learning_rate, epoch=epoch)
        writer.add_scalars('Train/Metrics', {'focal_loss': focal_loss, 'dice_loss':dice_loss}, epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        writer.add_scalar('Train/Metrics/f1_score', f1_score, epoch)
        grid = images_display.join_image_batches(samples)
        writer.add_images('train_sample', grid, epoch, dataformats='HWC')
        
        val_loss, val_f1_score, val_focal_loss, val_dice_loss, samples = metrics(net, validationloader, epoch)
        writer.add_scalars('Validation/Metrics', {'focal_loss': val_focal_loss, 'dice_loss':val_dice_loss}, epoch)
        writer.add_scalar('Validation/Metrics/loss', val_loss, epoch)
        writer.add_scalar('Validation/Metrics/f1_score', val_f1_score, epoch)
        grid = images_display.join_image_batches(samples)
        writer.add_images('validation_sample', grid, epoch, dataformats='HWC')

        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric < val_f1_score:
            train_config.iteration_age = 0
            train_config.best_metric = val_f1_score
            print('Epoch {}. Saving model with metric: {}'.format(epoch, val_f1_score))
            torch.save(net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age+=1
            print('Epoch {} metric: {}'.format(epoch, val_f1_score))
        if train_config.iteration_age==lower_learning_period:
            train_config.learning_rate*=0.5
            train_config.iteration_age=0
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        train_config.epoch = epoch+1
        train_config.save(checkpoint_conf_path)
        torch.save(net, checkpoint_name_path)
        torch.save(net.state_dict(), checkpoint_name_path.replace('.pth', '_state_dict.pth'))
    print('Finished Training')
    return train_config.best_metric

def test(net, testloader, dataset_name,):
    model_dir_header = net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_name_path = os.path.join(chp_dir, '{}_checkpoints_final.pth'.format(dataset_name))
    net = torch.load(checkpoint_name_path)
    net.cuda()
    writer = SummaryWriter(os.path.join('logs', model_dir_header))
        
    test_loss, test_f1_score, test_focal_loss, test_dice_loss, samples = metrics(net, testloader, 0)
    writer.add_scalars('Test/Metrics', {'focal_loss': test_focal_loss, 'dice_loss':test_dice_loss}, 0)
    writer.add_scalar('Test/Metrics/loss', test_loss, 0)
    writer.add_scalar('Test/Metrics/f1_score', test_f1_score, 0)
    grid = images_display.join_image_batches(samples)
    writer.add_images('Test_sample', grid, 0, dataformats='HWC')