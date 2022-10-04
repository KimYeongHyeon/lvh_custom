import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torchvision.transforms import Lambda, Compose
import torchvision.models as models
from torch.cuda import amp

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

from .evaluation import distance_error, AverageMeter


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion, CFG):
    model.train()

    losses = AverageMeter()
    mean_distance_error = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, sample in pbar:
        optimizer.zero_grad()
        x = sample['data']
        y = sample['label']

        x = x.to(device).float()
        y = y.to(device).float()

        preds = model(x).float()
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))
        
        mean_distance_error.update(distance_error(preds.reshape(-1, CFG['num_classes'], 3).detach().cpu().numpy(), 
                                                  y.reshape(-1, CFG['num_classes'], 3).detach().cpu().numpy()), 
                                   x.size(0))

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{losses.avg:.4f}',
                         train_MDE=f'{mean_distance_error.avg:.4f}',
                         lr=f'{current_lr:.5f}',
                         )
    return losses.avg, mean_distance_error.avg



@torch.no_grad()
def valid_one_epoch(model, dataloader, device, criterion, CFG):
    model.eval()
    
    losses = AverageMeter()
    mean_distance_error = AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')    

    for step, sample in pbar:
        x = sample['data']
        y = sample['label']

        x = x.to(device).float()
        y = y.to(device).float()

        preds = model(x).float()
        loss = criterion(preds, y)

        losses.update(loss.item(), x.size(0))

        mean_distance_error.update(distance_error(preds.reshape(-1, CFG['num_classes'], 3).detach().cpu().numpy(),
                                                  y.reshape(-1, CFG['num_classes'], 3).detach().cpu().numpy()), 
                                    x.size(0))

        pbar.set_postfix(valid_loss=f'{losses.avg:.4f}',
                         valid_MDE=f'{mean_distance_error.avg:.4f}')

    return losses.avg, mean_distance_error.avg
