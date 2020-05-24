import torch
from tqdm import tqdm
from functools import reduce
import numpy as np
from sklearn.metrics import f1_score
from model_fitting.losses import SegmentationLoss

def metrics(net, dataloader, box_transform, epoch=1):
    net.eval()
    criterion = SegmentationLoss()
    losses = 0.0
    total_focal_loss = 0.0
    total_dice_loss = 0.0
    f1_metric = 0.0
    images = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            image, labels = data
            outputs = net(image.cuda())
            loss, focal_loss, dice_loss = criterion(outputs, labels.cuda())
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