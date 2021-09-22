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
class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.file_names = df['file_path'].values
        self.ids = df['id'].values
    
    def __len__(self):
        return len(self.file_names)
    
    def cnn_1d_preprocess(self, waves):
        scaling = [1.5e-20, 1.5e-20, 0.5e-20]
        for i in range(3):
            waves[i] = waves[i] / scaling[i]
        return waves

    def bandpass_1dcnn(self, x, lf=20, hf=500, order=8, sr=2048):
        '''
        Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        '''
        sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
        normalization = np.sqrt((hf - lf) / (sr / 2))
        window = signal.tukey(4096, 0.1)
        if x.ndim ==2:
            x *= window
            for i in range(3):
                x[i] = signal.sosfilt(sos, x[i]) * normalization
        elif x.ndim == 3: # batch
            for i in range(x.shape[0]):
                x[i] *= window
                for j in range(3):
                    x[i, j] = signal.sosfilt(sos, x[i, j]) * normalization
        return x

    def __getitem__(self, ind):
        waves = np.load(self.file_names[ind])
        # image = self.apply_qtransformv2(waves, self.wave_transform)
        waves = self.cnn_1d_preprocess(waves)
        waves = self.bandpass_1dcnn(waves, lf = 30, hf = 1023, order=16)
        image = torch.from_numpy(waves).float()
        return image, self.ids[ind]


class TrainDataset(Dataset):
    def __init__(self, df, lf, hf, order = 8, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['target'].values
        self.wave_transform = CQT1992v2(sr = 2048, fmin = 20, fmax = 500, hop_length = 16, bins_per_octave=12)
        self.transform = transform
        self.fs = 4096
        self.fband = [lf, hf]
        self.dt = 0.000244141
        self.order = order
        
    def __len__(self):
        return len(self.labels)

    def apply_qtransform(self, waves):
        # waves = np.hstack(waves)
        waves = waves / np.max(waves) #np.max((np.max(waves), np.abs(np.min(waves))))
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
        bHP, aHP = butter(8, (20, 500), btype='bandpass', fs= 2048)
        strain_bp = filtfilt(bHP, aHP, strain)
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

            # whiten_strain = self.whiten(strain, psd, self.dt)
            band_strain = self.bandpass(strain, self.fband, self.fs)
            
            image.append(self.apply_qtransform(band_strain))
        image = torch.cat(image, dim=0)
        return image
    
    def apply_qtransformv2(self, waves, transform):
        image = self.whiten_bandpass(waves)
        image = image.permute(1, 2, 0)
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        image = image.squeeze().numpy()
        return image
    
    def cnn_1d_preprocess(self, waves):
        scaling = [1.5e-20, 1.5e-20, 0.5e-20]
        for i in range(3):
            waves[i] = waves[i] / scaling[i]
        return waves

    def bandpass_1dcnn(self, x, lf=20, hf=500, order=8, sr=2048):
        '''
        Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        '''
        sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
        normalization = np.sqrt((hf - lf) / (sr / 2))
        window = signal.tukey(4096, 0.1)
        if x.ndim ==2:
            x *= window
            for i in range(3):
                x[i] = signal.sosfilt(sos, x[i]) * normalization
        elif x.ndim == 3: # batch
            for i in range(x.shape[0]):
                x[i] *= window
                for j in range(3):
                    x[i, j] = signal.sosfilt(sos, x[i, j]) * normalization
        return x

    def __getitem__(self, ind):
        waves = np.load(self.file_names[ind])
        # image = self.apply_qtransformv2(waves, self.wave_transform)
        waves = self.cnn_1d_preprocess(waves)
        waves = self.bandpass_1dcnn(waves, lf = self.fband[0], hf = self.fband[1], order = self.order)
        image = torch.from_numpy(waves).float()
        if self.transform:
            image = self.transform(image = image)['image']
        label = torch.tensor(self.labels[ind]).float()
        return image, label