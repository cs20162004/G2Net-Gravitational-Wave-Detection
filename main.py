import time
from utils import AverageMeter, timeSince, get_score
from dataset.datasets import TrainDataset, get_transforms
from model.models import Efficientnet7
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()
    start = end = time.time()
    global_step = 0
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

        optimizer.zero_grad()
        global_step += 1
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 100 == 0 or step == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
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
def train_loop(folds, fold):
    
    print(f"========== fold: {fold} training ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['target'].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=64, 
                              shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=64 * 2, 
                              shuffle=False, 
                              num_workers=4, pin_memory=True, drop_last=False)
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = Efficientnet7(pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(3):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        "../weights/"+f'EfficientNet7_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        "../weights/"+f'EfficientNet7_fold{fold}_best_loss.pth')
    
    valid_folds['preds'] = torch.load("../weights/"+f'EfficientNet7_fold{fold}_best_score.pth', 
                                      map_location=torch.device('cpu'))['preds']

    return valid_folds



def main():

    """
    Prepare: 1.train 
    """
    train = pd.read_csv("/home/hero/Downloads/Ali/practice/g2net/dt/split_train.csv")

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df['target'].values
        score = get_score(labels, preds)
        print(f'Score: {score:<.4f}')
    
    # train 
    oof_df = pd.DataFrame()
    for fold in range(5):
        if fold in [0]:
            _oof_df = train_loop(train, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            print(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    # CV result
    print(f"========== CV ==========")
    get_result(oof_df)
    # save result
    oof_df.to_csv("../weights/"+'oof_df.csv', index=False)

if __name__ == '__main__':
    main()