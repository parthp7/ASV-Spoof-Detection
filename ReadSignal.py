#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 01:23:31 2018

@author: parth
"""
import scipy.io.wavfile as wavreader
from scipy import signal
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # Reading wav file, returns tuple with framerate and speech sample array
    (frame_rate, speech_signal) = wavreader.read('./DemoSounds/noisy-signal.wav')
    
    speech_signal = speech_signal[:10000]
    
    print("Frame rate: {0}\nTotal samples: {1}".format(frame_rate, len(speech_signal)))
    plt.figure(1)
    plt.plot(speech_signal)
    
    # Pre-emphasis filter on speech signal
    # y[i] = x[i] - a*x[i-1]
    
    pre_emphasis = [0]
    alpha = 0.97            # filter coefficient for pre-emphasis
    
    for i in range(1,len(speech_signal)):
        pre_emphasis.append(speech_signal[i]-alpha*speech_signal[i-1])
        
    # Plot for pre-emphasis filtered signal
    plt.figure(2)
    plt.plot(pre_emphasis)
    
    # Applying Hamming window function
    # Window length 256 and 50% overlap
    
    win_length = 256
    window = signal.hamming(win_length, False)
    
    plt.figure(3)
    plt.plot(window)