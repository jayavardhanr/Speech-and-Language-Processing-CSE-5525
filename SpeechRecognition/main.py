#!/usr/bin/env python
from __future__ import division

import argparse
import math

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

from scipy import fftpack
from scipy import signal

def power(spec):
    return spec.real ** 2 + spec.imag ** 2

def mel(f):
    return 1127 * math.log(1 + f / 700)

def inv_mel(m):
    return 700 * (math.exp(m / 1127) - 1)

def euclidean(a, b):
    return np.linalg.norm(a - b)

def _log_spectrum(window, sample_freq):
    """
    Args:
    window: Time domain window.
    sample_freq: The frequency of samples in the original data.
    """
    hamming = np.hamming(len(window))
    spectrum = np.fft.fft(window * hamming)
    power_spectrum = power(spectrum)

    linear_cutoff = 1000
    unit = len(window) / sample_freq

    linear_bounds = np.arange(0, linear_cutoff + 1, 100) * unit

    log_max = mel(sample_freq / 2)
    log_min = mel(linear_cutoff)
    log_mel_bounds = np.linspace(log_min, log_max, num=18)
    log_bounds = np.zeros_like(log_mel_bounds)

    for i, el in enumerate(log_mel_bounds):
        log_bounds[i] = inv_mel(el) * unit

    all_bounds = np.concatenate([linear_bounds, log_bounds[1:]])
    mel_filterbanks = []
    for i in range(len(all_bounds) - 2):
        upper = int(all_bounds[i + 2])
        lower = int(all_bounds[i])

        filter_bank = signal.triang(upper - lower)
        power_spec = filter_bank * power_spectrum[lower:upper]
        total_power = np.sum(np.log(power_spec))

        mel_filterbanks.append(total_power)

    return np.array(mel_filterbanks)

def log_spectrum(window, sample_freq):
    mel_filterbanks = _log_spectrum(window, sample_freq)
    return mel_filterbanks

def mfcc(window, sample_freq):
    """
    Args:
    window: Time domain window.
    sample_freq: The frequency of samples in the original data.
    """
    mel_filterbanks = _log_spectrum(window, sample_freq)
    cepstrum = scipy.fftpack.dct(mel_filterbanks)
    cepstrals = cepstrum[:12]
    return cepstrals

def dtw(x, y, dist):
    l_x = len(x)
    l_y = len(y)

    d = np.zeros((l_x, l_y), dtype=float)
    b = np.zeros((l_x, l_y, 2), dtype=int)
    d[0][0] = dist(x[0], y[0])

    for i in range(1, l_x):
        d[i][0] = dist(x[i], y[0]) + d[i - 1][0]
        b[i][0][0] = -1

    for j in range(1, l_y):
        d[0][j] = dist(y[j], x[0]) + d[0][j - 1]
        b[0][j][1] = -1

    for i in range(1, l_x):
        for j in range(1, l_y):
            paths = (d[i - 1][j], d[i - 1][j - 1], d[i][j - 1])
            d[i][j] = min(paths) + dist(x[i], y[j])

            am = np.argmin(paths)
            if am == 0:
                b[i][j][0] = -1
            elif am == 1:
                b[i][j][0] = -1
                b[i][j][1] = -1
            else:
                b[i][j][1] = -1

    return d[l_x - 1][l_y - 1]

def delta(feat, N):
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
    return delta_feat

def _get_transformed_vector(x, sf, winsize, shift, transform):
    extra = np.zeros((len(x) - 512) % shift)
    x = np.concatenate([x, extra])

    shifts = int((len(x) - 512) / shift)
    mfcc_vectors = []
    for i in range(shifts):
        start = shift * i
        vec = transform(x[start:start + winsize], sf)
        mfcc_vectors.append(vec)

    return mfcc_vectors
    
def get_log_spectrum_vectors(x, sf, winsize, shift):
    return _get_transformed_vector(x, sf, winsize, shift, log_spectrum)

def get_mfcc_vectors(x, sf, winsize, shift):
    features=_get_transformed_vector(x, sf, winsize, shift, mfcc)

    ##Cepstral features
    features1=np.array(features)
    delta_features_cep=delta(features1,2)
    double_delta_features_cep=delta(delta_features_cep,2)
    
    concatenated=np.concatenate((features1,delta_features_cep,double_delta_features_cep),axis=1)
    return concatenated

parser = argparse.ArgumentParser()
parser.add_argument('--templates', nargs='+', required=True,
    help='The files to use as templates')
parser.add_argument('--predict', nargs='+', required=True,
    help='The files to classify')
parser.add_argument('--transformation', type=int, required=True,
    help='0 for MFCC and 1 for log spectrum')
args = parser.parse_args()

winsize = 512
shift = 20

if args.transformation == 0:
    vector_transform = get_mfcc_vectors
elif args.transformation == 1:
    vector_transform = get_log_spectrum_vectors

vector_representations = {}
for fn in args.templates:
    (sf, x) = scipy.io.wavfile.read(fn)
    representation = vector_transform(x, sf, winsize, shift)
    vector_representations[fn] = representation
    print('Done processing {}'.format(fn))

print('Prediction...')
for predict_fn in args.predict:
    (sf_predict, x_predict) = scipy.io.wavfile.read(predict_fn)
    representation_predict = vector_transform(x_predict, sf_predict, winsize, shift)
    min_cost = float('inf')
    for fn, vectors in vector_representations.items():
        print('Evaluating {}...'.format(fn))
        cost = dtw(representation_predict, vectors, euclidean)
        if cost < min_cost:
            min_cost = cost
            min_template = fn

    print('*********{} is predicted as template {}*********'.format(min_template, predict_fn))
