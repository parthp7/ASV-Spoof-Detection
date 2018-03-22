#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 01:23:31 2018

@author: parth
"""
import scipy.io.wavfile as wavreader
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    #Reading wav file, returns tuple with framerate and speech sample array
    (frame_rate, speech_signal) = wavreader.read('./DemoSounds/brian.wav')
    
    print(frame_rate)
    plt.plot(speech_signal)    