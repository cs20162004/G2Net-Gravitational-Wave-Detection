import time
from utils import AverageMeter, timeSince, get_score, Logger
from dataset.datasets import TrainDataset, get_transforms
from model.models import Efficientnet7, Efficientnetv2_b1
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import argparse

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    start = end = time.time()
    for step, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)

        losses.update(loss.item(), batch_size)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1000)
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 100 == 0 or step == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}'
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % 100 == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions



# ====================================================
# Train loop
# ====================================================
def train_loop(args, folds, fold, size, gpu, total_gpus, logger):
    if gpu == 0:    print(f"========== fold: {fold} training ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['target'].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train', size=size))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='train', size=size))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = total_gpus, rank = gpu)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, 
                            #   shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler = train_sampler)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=args.batch_size * 2, 
                              shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = Efficientnetv2_b1(pretrained=True)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(args.num_epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        if gpu == 0:
            print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            logger.write(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            if gpu == 0:
                print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
                torch.save({'model': model.state_dict(), 
                            'preds': preds},
                            os.path.join(args.out_dir, f'{args.model}_fold{fold}_best_score.pth'))
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if gpu == 0:
                print(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict(), 
                            'preds': preds},
                            os.path.join(args.out_dir, f'{args.model}_fold{fold}_best_loss.pth'))

    return best_score



def main(gpu, total_gpus, args):
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://', world_size = total_gpus, rank = gpu)

    # initialize logger and read csv file
    logger = Logger(os.path.join(args.out_dir, "train.log"))
    train = pd.read_csv(args.train_csv_path)

    # train 
    oof_df = pd.DataFrame()
    for size in args.image_size:
        if gpu == 0:    logger.write(f"=============== Size: {size}  ===============")
        for fold in range(args.num_folds):
            if fold in args.fold_list:
                best_score = train_loop(args, train, fold, size, gpu, total_gpus, logger)
                if gpu == 0:    
                    logger.write(f"========== fold: {fold} result ==========")
                    logger.write(f'Score: {best_score:<.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default = [384], type = list, help = "image size to train on (default: 256)")
    parser.add_argument('--out_dir', default = '../weights', type = str, help = "log file to save training result")
    parser.add_argument('--local_rank', type = int, default=0)
    parser.add_argument('--loss', type = str, default = 'BCEWithLogitsLoss', help = "loss function (default: BCEWithLogitsLoss")
    parser.add_argument('--num_workers', type = int, default=4, help = "number of data loading workers (default: 4)")
    parser.add_argument('--batch_size', type = int, default = 64, help = "batch size (default: 64)")
    parser.add_argument('--in_channels', type = int, default = 3, help = "input channels (default: 3)")
    parser.add_argument('--train_csv_path', type = str, default = "../dt/split_train.csv", help = "training csv path")
    parser.add_argument('--num_folds', type = int, default = 5, help = "total number of folds")
    parser.add_argument('--fold_list', type = list, default = [0,1,2,3,4], help="fold list to train on")
    parser.add_argument('--num_epochs', type = int, default = 5, help = "number of epochs (default: 5)")
    parser.add_argument('--model', type = str, default = "Efficientnetv2_b1", help = "model architecture to train") 
    args = parser.parse_args()

    total_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.1.1'
    os.environ['MASTER_PORT'] = '1234'
    torch.multiprocessing.spawn(main, nprocs = total_gpus, args = (total_gpus, args))