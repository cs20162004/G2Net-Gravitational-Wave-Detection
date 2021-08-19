from torch.utils.data import Dataset
import torch
import numpy as np
from nnAudio.Spectrogram import CQT1992v2
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import Resize
import cv2


def get_transforms(*, data, size):
    
    if data == 'train':
        return A.Compose([
            A.Resize(size, size, cv2.INTER_CUBIC),
            ToTensorV2(),

        ])

    elif data == 'valid':
        return A.Compose([
            Resize(size, size, cv2.INTER_CUBIC),
            ToTensorV2(),
        ])

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['target'].values
        self.wave_transform = CQT1992v2(sr = 4096, fmin = 20, fmax = 1024, hop_length = 64, bins_per_octave = 8)
        self.transform = transform
        self.fs = 4096
        self.fband = [20.0, 500.0]
        self.dt = 0.000244141

        
    def __len__(self):
        return len(self.labels)

    def apply_qtransform(self, waves):
        # waves = np.hstack(waves)
        # waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = self.wave_transform(waves)
        return image
    
    # function to whiten data
    def whiten(self, strain, interp_psd, dt):
        Nt = len(strain)
        freqs = np.fft.rfftfreq(Nt, dt)
        # freqs1 = np.linspace(0,2048.,Nt/2+1)

        # whitening: transform to freq domain, divide by asd, then transform back, 
        # taking care to get normalization right.
        hf = np.fft.rfft(strain)
        norm = 1./np.sqrt(1./(dt*2))
        white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
        white_ht = np.fft.irfft(white_hf, n=Nt)
        return white_ht


    def bandpass(self, strain, fband, fs):
        bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
        normalization = np.sqrt((fband[1] - fband[0])/(fs/2))
        strain_bp = filtfilt(bb, ab, strain) / normalization
        return strain_bp

    def whiten_bandpass(self, waves):
        image = []
        for strain in waves:
            # 1. windowing
            tukey_window = signal.tukey(self.fs, 0.125)
            strain = strain * tukey_window
            psd_window = signal.tukey(2*self.fs, alpha=1./4)
            
            # number of samples for fft
            NFFT=2 * self.fs
            Pxx, freqs = mlab.psd(strain, Fs = self.fs, NFFT = NFFT, window=psd_window, noverlap=4096)
            psd = interp1d(freqs, Pxx)

            whiten_strain = self.whiten(strain, psd, self.dt)
            band_strain = self.bandpass(whiten_strain, self.fband, self.fs)

            image.append(self.apply_qtransform(band_strain))
        image = torch.cat(image, dim=0)
        return image

    def __getitem__(self, ind):
        waves = np.load(self.file_names[ind])
        image = self.whiten_bandpass(waves)
        image = image.permute(1, 2, 0)
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
