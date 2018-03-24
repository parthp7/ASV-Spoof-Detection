#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 21:53:56 2018

@author: parth
"""

import scipy.io.wavfile as wavreader
import matplotlib.pyplot as plt
import SignalProcessing as sigproc
import numpy as np
from scipy.fftpack import dct


def mfcc(signal, samplerate, winlen, winstep, numcep=12, nfft=256, nfilt=26, lowfreq=0, highfreq=None, appendEnergy=True):
    ''' MFCC features of an audio signal '''
    features, energy = mel_filterbank(signal, samplerate, winlen, winstep, nfft, nfilt, lowfreq, highfreq)
    features = np.log(features)
    features = dct(features, type=2, axis=1, norm='ortho')[:,:numcep]
    features = lifter(features)
    if appendEnergy: features[:,0] = np.log(energy)
    return features


def mel_filterbank(signal, samplerate, winlen, winstep, nfft=256, nfilt=26, lowfreq=0, highfreq=None):
    ''' Computes Mel-filterbank energy features from audio signal '''
    
    signal = sigproc.pre_emphasis_filter(signal)
    frames = sigproc.window_and_overlap(signal, winlen, winstep)
    powspec = sigproc.power_spectrum(frames, nfft)
    energy = np.sum(powspec, 1)
    energy = np.where(energy==0, np.finfo(float).eps, energy)
    
    fbank = get_filterbank(nfilt, nfft, samplerate)
    features = np.dot(powspec, fbank.T)
    features = np.where(features==0, np.finfo(float).eps, features)
    
    return features, energy


def hz2mel(hz):
    """Convert a value in Hertz to Mels """
    return 2595 * np.log10(1+hz/700.0)

def mel2hz(mel):
    """Convert a value in Mels to Hertz """
    return 700*(10**(mel/2595.0)-1)


def get_filterbank(nfilt, nfft, samplerate, lowfreq=0, highfreq=None):
    ''' Compute a Mel-filterbank. Filters are stored in rows, columns are fft bins.
        Returns an array of size nfilt x (nfft/2 + 1) '''
    
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    
    # Evenly spaced points in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    
    # Converting mel points to fft bins by first converting them back to Hz
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)
    
    # Filterbank of size nfilt x nfft/2+1
    fbank = np.zeros([nfilt,nfft//2+1])
    
    # Placing triangular filterbanks in fbank
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the magnitude of the high frequency DCT coeffs."""
    if L>0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.0)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        return cepstra


if __name__ == "__main__":
    
    # Reading wav file, returns tuple with sample rate and speech sample array
    (sample_rate, speech_signal) = wavreader.read('./DemoSounds/noisy-signal.wav')
    
    speech_signal = speech_signal[:10000]
    
    print("Frame rate: {0}\nTotal samples: {1}".format(sample_rate, len(speech_signal)))
    
    plt.plot(mfcc(speech_signal, sample_rate, 256, 128))