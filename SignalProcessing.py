#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 01:23:31 2018

@author: parth
"""
import scipy.io.wavfile as wavreader
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
import logging


def pre_emphasis_filter(signal, filter_coeff=0.97):
    """ Returns list containing pre-emphasized signal using formula: y[i] = x[i] - a*x[i-1] """
    return np.append(signal[0],signal[1:]-filter_coeff*signal[:-1])


def rolling_window(sig, window, step):
    """ Returns 2-D array from 1-D sig by appling 'window' and 'step' on sig """
    shape = sig.shape[:-1] + (sig.shape[-1] - window + 1, window)
    strides = sig.strides + (sig.strides[-1],)
    return np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)[::step]


def hamming_window(length):
    """ Returns hamming window function with periodic window"""
    return signal.hamming(length, False)


def window_and_overlap(signal, frame_len, frame_step, winfunc=hamming_window):
    """ Apply window function to divide signal into frames """
    slen = len(signal)
    frame_len = int(frame_len)
    frame_step = int(frame_step)
    
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((slen-frame_len)/frame_step))
        
    padlen = (numframes-1)*frame_step + frame_len
    zeros = np.zeros((padlen-slen,))
    padsig = np.concatenate((signal, zeros))
    
    window = winfunc(frame_len)
    frames = rolling_window(padsig, frame_len, frame_step)
    
    return frames*window


def magnitude_spectrum(frames, NFFT):
    """ Magnitude spectrum for each frame """
    
    if np.shape(frames)[1] > NFFT:
        logging.warn('frame length is greater than NFFT, increase NFFT.')
    
    complex_spectrum = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spectrum)


def phase_spectrum(frames, NFFT):
    """ Phase spectrum for each frame """
    if np.shape(frames)[1] > NFFT:
        logging.warn('frame length is greater than NFFT, increase NFFT.')
    
    complex_spectrum = np.fft.rfft(frames, NFFT)
    return np.unwrap(np.angle(complex_spectrum))


def power_spectrum(frames, NFFT):
    """ Computes the power spectrum for each frame of windowed signal
        NFFT is the FFT length to use. The execution time for fft depends 
        on the length of the transform. It is fastest for powers of two. 
        It is almost as fast for lengths that have only small prime factors.
    :returns: for NxD matrix, outputs Nx(NFFT/2 + 1). Each row is power spectrum for corresponding frame
    """
    return 1.0/NFFT * np.square(magnitude_spectrum(frames, NFFT))


def log_power_spectrum(frames, NFFT, normalized = True):
    pow_spec = power_spectrum(frames, NFFT)
    pow_spec[pow_spec<=1e-30] = 1e-30
    
    log_pow_spec = 10*np.log10(pow_spec)
    if normalized:
        return log_pow_spec - np.max(log_pow_spec)
    else:
        return log_pow_spec

"""
if __name__ == '__main__':
    
    # Reading wav file, returns tuple with framerate and speech sample array
    (frame_rate, speech_signal) = wavreader.read('./DemoSounds/noisy-signal.wav')
    
    speech_signal = speech_signal[:10000]
    
    print("Frame rate: {0}\nTotal samples: {1}".format(frame_rate, len(speech_signal)))
    plt.figure(1)
    plt.plot(speech_signal)
    
    # Pre-emphasis filter on speech signal
    pre_emphasis = pre_emphasis_filter(speech_signal)
        
    # Plot for pre-emphasis filtered signal
    plt.figure(2)
    plt.plot(pre_emphasis)
    
    # Applying Hamming window function
    # Window length 256 and 50% overlap
    window_len = 256
    windowed_signal = window_and_overlap(pre_emphasis, window_len, window_len/2)
    
    pow_spec = power_spectrum(windowed_signal, 256)
    energy = np.sum(pow_spec, 1)
    
    plt.figure(3)
    plt.plot(energy)
    """