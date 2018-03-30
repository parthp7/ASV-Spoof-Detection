#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:38:49 2018

@author: parth
"""

import scipy.io.wavfile as wavreader
import FeatureExtraction as feat
import numpy as np
import time

DIR_PATH = "/media/parth/Entertainment/ASV2015/"


def file_read(filename):
    
    wav_files = list()
    
    with open("{}{}".format(DIR_PATH, filename), "r") as file:
        for line in file:
            if line == "": break
            record = line.split()
            #if record[0] == "T1" and record[3] == "human":
            wav_files.append(record)
    
    return wav_files


def feature_extraction_for_each(wav_files, feature="mfcc"):
    
    for file in wav_files:
        (sample_rate, speech_signal) = wavreader.read("{}wav/{}/{}.wav".format(DIR_PATH, file[0], file[1]))
        
        feature_vector = feat.mfcc(speech_signal, sample_rate, 0.016, 0.008)
        
        derivative1 = feat.delta(feature_vector)
        derivative2 = feat.delta(derivative1)
        
        feature_vector = np.append(feature_vector, derivative1, axis=1)
        feature_vector = np.append(feature_vector, derivative2, axis=1)
        
    


if __name__ == "__main__":
    
    start_time = time.time()
    wav_files = file_read("CM_protocol/cm_train.trn")
    feature_extraction_for_each(wav_files)
    end_time = time.time()
    print(end_time-start_time)