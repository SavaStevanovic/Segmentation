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
import numpy as np

def fit_epoch(net, dataloader, lr_rate, epoch=1):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr_rate)
    criterion = torch.nn.CrossEntropyLoss()
    losses = 0.0
    f1_metric = 0.0
    images = None
    for i, data in enumerate(tqdm(dataloader)):
        image, labels = data
        optimizer.zero_grad()
        outputs = net(image.cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        losses += loss.item()
        outputs = outputs.softmax(1).detach().cpu().numpy()
        output_threshold = (outputs[:, 1] > 0.5).astype('float')
        f1_metric += f1_score(output_threshold.flatten(), labels.flatten())

        if i>=len(dataloader)-1:
            images = (( image.numpy(), 
                        outputs[:, 1], 
                        output_threshold,
                        labels.numpy()))
        
    data_len = len(dataloader)
    return losses/data_len, f1_metric/data_len, images

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
        loss, f1_score, samples = fit_epoch(net, trainloader, train_config.learning_rate, epoch=epoch)
        writer.add_scalar('Train/Metrics/loss', loss, epoch)
        writer.add_scalar('Train/Metrics/f1_score', f1_score, epoch)
        grid = images_display.join_images(samples)
        writer.add_images('train_sample', grid, epoch, dataformats='CHW')
        
        val_loss, val_f1_score, samples = metrics(net, validationloader, epoch)
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
    return best_map