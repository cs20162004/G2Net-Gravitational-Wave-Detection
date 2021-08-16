from torch.utils.data import Dataset
import torch
import numpy as np
from nnAudio.Spectrogram import CQT1992v2
import albumentations as A 
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            ToTensorV2(),
        ])

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['target'].values
        self.wave_transform = CQT1992v2(sr = 4096, fmin = 20, fmax = 1024, hop_length = 32, bins_per_octave = 8)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)

    def apply_qtransform(self, waves, transform):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def __getitem__(self, ind):
        waves = np.load(self.file_names[ind])
        image = self.apply_qtransform(waves, self.wave_transform)
        image = image.squeeze().numpy()
        if self.transform:
            image = self.transform(image = image)['image']
        label = torch.tensor(self.labels[ind]).float()
        return image, label


class GradCAMDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.image_ids = df['id'].values
        self.file_names = df['file_path'].values
        self.labels = df['target'].values
        self.wave_transform = CQT1992v2(sr = 4096, fmin = 20, fmax = 1024, hop_length = 32, bins_per_octave = 8)
        self.transform = get_transforms(data='valid')
        
    def __len__(self):
        return len(self.df)
    
    def apply_qtransform(self, waves, transform):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_path = self.file_names[idx]
        waves = np.load(file_path)
        image = self.apply_qtransform(waves, self.wave_transform)
        image = image.squeeze().numpy()
        vis_image = image.copy()
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx]).float()
        return image_id, image, vis_image, label