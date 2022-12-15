
##### HifiGan Based

## Audio
sample_rate = 22050

## Mel-filterbank
n_fft = 1024
num_mels = 80
num_samples = 128 # input spect shape (num_mels, num_samples)
hop_length = 256
win_length = 1024
fmin = 0
fmax = 8000

max_min_norm = True
min_value = -11.6 # Female - Male
max_value = 3.1 # Female - Male

# min_value = -11.512925148010254 # Male - Male
# max_value = 2.175222873687744 # Male - Male

MAX_WAV_VALUE = 32768.0

###### Original

## Audio
# sample_rate = 16000

# ## Mel-filterbank
# n_fft = 2048
# num_mels = 128
# num_samples = 128 # input spect shape (num_mels, num_samples)
# hop_length = int(0.0125*sample_rate)
# win_length = int(0.05*sample_rate)
# fmin = 40
# fmax = sample_rate // 2

# MAX_WAV_VALUE = 32768.0