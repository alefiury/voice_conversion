import argparse
import pickle

import torch
import numpy as np
from tqdm import tqdm

from params import *
from utils import ls, preprocess_wav
from melspectrogram_extractor import mel_spectrogram


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--n_spkrs", type=int, default=2, help="size of the batches")

opt = parser.parse_args()
print(opt)
feats = {}

for spkr in range(opt.n_spkrs):
    wavs = ls('%s/spkr_%s | grep .wav'%(opt.dataset, spkr+1))
    feats[spkr] = [None]*len(wavs)
    for i, wav in tqdm(enumerate(wavs), total=len(wavs), desc="spkr_%d"%(spkr+1)):
        sample = preprocess_wav('%s/spkr_%s/%s'%(opt.dataset, spkr+1, wav))
        y = torch.FloatTensor(sample.astype(np.float32))
        y = y.unsqueeze(0)
        feats[spkr][i] = mel_spectrogram(
            y=y,
            n_fft=n_fft,
            num_mels=num_mels,
            sampling_rate=sample_rate,
            hop_size=hop_length,
            win_size=win_length,
            fmin=fmin,
            fmax=fmax,
            center=False
        )

pickle.dump(feats,open('%s/%s.pickle'%(opt.dataset, opt.model_name),'wb'))
