import argparse
from dataset.datasets import TestDataset
from model.models import Efficientnet7, Efficientnetv2_b1, Efficientnet_b0, cnn_1d
import torch
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == "cnn_1d":
        model = cnn_1d()
    model.to(device)
    
    test_csv = pd.read_csv("../dt/test_sample.csv")
    test_dataset = TestDataset(test_csv)

    test_loader = DataLoader(test_dataset, 
                              batch_size=args.batch_size * 2, 
                              shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    probs_all = []
    wave_ids = []
    k = 0
    for fold in args.num_folds:
        weights = torch.load(f"../cnn_1d_256_0.005_lf30_hf1023_order16/cnn_1d_size_256_fold{fold}_best_score.pth")['model']
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        probs = []

        for data, id_ind in tqdm(test_loader):
            data = data.to(device)
            with torch.no_grad():
                y_preds = model(data)
            preds = y_preds.sigmoid().to('cpu').numpy()
            probs.append(preds)
            if k==0:
                wave_ids.append(id_ind)
        k = 1

        probs = np.concatenate(probs)
        probs_all.append(probs)

    probs_avg = np.asarray(probs_all).mean(axis=0).flatten()
    wave_ids = np.concatenate(wave_ids)

    test_df = pd.DataFrame({'id': wave_ids, 'target': probs_avg})
    # Save test dataframe to disk
    test_df.to_csv(f'{args.model}_test_result.csv', index = False)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = "cnn_1d", help = "model name")
    parser.add_argument('--num_folds', type = list, default = [1,2,3,4], help = "number of folds")
    parser.add_argument('--num_workers', type = int, default = 2, help = "number of workers")
    parser.add_argument('--batch_size', type = int, default = 64, help = "batch size")
    args = parser.parse_args()

    main(args)