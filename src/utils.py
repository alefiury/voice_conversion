# SOURCE:
# - https://github.com/CorentinJ/Real-Time-Voice-Cloning
# - https://github.com/r9y9/wavenet_vocoder

from scipy.ndimage.morphology import binary_dilation
import os
import math
import numpy as np
from pathlib import Path
from typing import Optional, Union
import librosa
import struct
from params import *
from scipy.signal import lfilter

try:
    import webrtcvad
except:
    warn("Unable to import 'webrtcvad'. This package enables noise removal and is recommended.")
    webrtcvad=None

int16_max = (2 ** 15) - 1

def preprocess_wav(
    fpath_or_wav: Union[str, Path, np.ndarray],
    source_sr: Optional[int] = None
):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sample_rate:
        wav = librosa.resample(wav, source_sr, sample_rate)

    return wav


def ls(path):
    return os.popen('ls %s'%path).read().split('\n')[:-1]

def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)


def load_wav(path):
    return librosa.load(path, sr=sample_rate)[0]


def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=sample_rate)


def split_signal(x):
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2**15


def encode_16bits(x):
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin)


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)



def stft(y):
    return librosa.stft(
        y=y,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def pre_emphasis(x):
    return lfilter([1, -preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -preemphasis], x)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels: y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

# def reconstruct_waveform(mel, n_iter=32):
#     """Uses Griffin-Lim phase reconstruction to convert from a normalized
#     mel spectrogram back into a waveform."""
#     denormalized = denormalize(mel)
#     amp_mel = db_to_amp(denormalized)
#     S = librosa.feature.inverse.mel_to_stft(
#         amp_mel, power=1, sr=sample_rate,
#         n_fft=n_fft, fmin=fmin)
#     wav = librosa.core.griffinlim(
#         S, n_iter=n_iter,
#         hop_length=hop_length, win_length=win_length)
#     return wav
