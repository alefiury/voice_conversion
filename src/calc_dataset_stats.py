import argparse
import pickle

import torch
import numpy as np
from tqdm import tqdm

from params import *
from utils import ls, preprocess_wav
from melspectrogram_extractor import mel_spectrogram

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to dataset")
    parser.add_argument("--n_spkrs", type=int, default=2, help="size of the batches")
    opt = parser.parse_args()

    maxs = []
    mins = []

    for spkr in range(opt.n_spkrs):
        wavs = ls('%s/spkr_%s | grep .wav'%(opt.dataset, spkr+1))
        for i, wav in tqdm(enumerate(wavs), total=len(wavs), desc="spkr_%d"%(spkr+1)):
            sample = preprocess_wav('%s/spkr_%s/%s'%(opt.dataset, spkr+1, wav))
            y = torch.FloatTensor(sample)
            y /= torch.abs(y).max()

            y = y.unsqueeze(0)
            mel_spec = mel_spectrogram(
                y=y,
                n_fft=n_fft,
                num_mels=num_mels,
                sampling_rate=sample_rate,
                hop_size=hop_length,
                win_size=win_length,
                fmin=fmin,
                fmax=fmax,
                center=False,
                max_value=0.0,
                min_value=0.0,
                max_min_norm=False,
            ).squeeze(0)

            maxs.append(mel_spec.max())
            mins.append(mel_spec.min())

    print(f"Max: {max(maxs)} | Min: {min(mins)}")


if __name__ == "__main__":
    main()