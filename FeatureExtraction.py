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


def mel_filterbank(signal, samplerate, winlen, winstep, nftt=256):
    ''' Computes Mel-filterbank energy features from audio signal '''
    
    signal = sigproc.pre_emphasis_filter(signal)
    frames = sigproc.window_and_overlap(signal, winlen, winstep)
    powspec = sigproc.power_spectrum(frames, nftt)
    energy = np.sum(powspec, 1)
    energy = np.where(energy==0, np.finfo(float).eps, energy)
    
    return energy


if __name__ == "__main__":
    
    # Reading wav file, returns tuple with sample rate and speech sample array
    (sample_rate, speech_signal) = wavreader.read('./DemoSounds/noisy-signal.wav')
    
    speech_signal = speech_signal[:10000]
    
    print("Frame rate: {0}\nTotal samples: {1}".format(sample_rate, len(speech_signal)))
    plt.figure(1)
    plt.plot(mel_filterbank(speech_signal, 8000, 256, 128))